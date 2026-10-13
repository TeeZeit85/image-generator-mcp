import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import { FALLBACK_CHAIN } from "./models.js";
import { loadKeyFiles, KEY_FILE } from "./config.js";
import { OpenAIError } from "./openai.js";

/**
 * OpenRouter backend (IMAGE_PROVIDER=openrouter).
 *
 * OpenRouter has one image endpoint, POST /images, that takes reference images
 * as data URLs instead of a multipart upload. Every model advertises its
 * `supported_parameters`, so params are filtered or translated per model
 * (e.g. gpt-image there takes aspect_ratio, not WIDTHxHEIGHT).
 */

loadKeyFiles();

const API_BASE = process.env.OPENROUTER_BASE_URL?.replace(/\/+$/, "") || "https://openrouter.ai/api/v1";

function apiKey() {
  const key = process.env.OPENROUTER_API_KEY?.trim();
  if (!key) {
    throw new Error(
      `OPENROUTER_API_KEY is not set. Either write it to ${KEY_FILE} as ` +
        "OPENROUTER_API_KEY=sk-or-... , or pass it through the MCP server config."
    );
  }
  return key;
}

async function request(path, { method = "POST", json } = {}) {
  const headers = { Authorization: `Bearer ${apiKey()}`, "X-Title": "image-generator-mcp" };
  let body;
  if (json) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  }

  const res = await fetch(`${API_BASE}${path}`, { method, headers, body });
  const text = await res.text();
  let parsed;
  try {
    parsed = text ? JSON.parse(text) : {};
  } catch {
    if (!res.ok) throw new OpenAIError(res.status, { error: { message: text.slice(0, 500) } }, "OpenRouter");
    throw new Error(`OpenRouter returned non-JSON response: ${text.slice(0, 200)}`);
  }
  if (!res.ok) throw new OpenAIError(res.status, parsed, "OpenRouter");
  return parsed;
}

/* ------------------------------------------------------------ model catalog */

let catalog;

/** Every image-output model on OpenRouter. Fetched once per server run. */
export async function imageModels() {
  catalog ??= request("/images/models", { method: "GET" })
    .then((res) => res.data || [])
    .catch((err) => {
      catalog = undefined;
      throw err;
    });
  return catalog;
}

const GPT_IMAGE = /^openai\/gpt-image-(\d+(?:\.\d+)?)/;

/** gpt-image models, newest first: higher version, then full before -mini, then release date. */
export function rankGptImage(models) {
  const version = (m) => parseFloat(m.id.match(GPT_IMAGE)[1]);
  const isMini = (m) => Number(/-mini\b/.test(m.id));
  return models
    .filter((m) => GPT_IMAGE.test(m.id))
    .sort(
      (a, b) =>
        version(b) - version(a) || isMini(a) - isMini(b) || (b.created || 0) - (a.created || 0)
    );
}

/**
 * Map what the caller typed onto an OpenRouter image model id. Bare OpenAI ids
 * (gpt-image-2) and unambiguous suffixes (flux.2-pro) are accepted; anything that
 * is not an image model is refused with the closest image-model matches.
 */
export async function resolveModel(id) {
  let models;
  try {
    models = await imageModels();
  } catch {
    return id.includes("/") ? id : `openai/${id}`;
  }
  if (models.some((m) => m.id === id)) return id;

  if (!id.includes("/")) {
    const exactOpenAI = models.find((m) => m.id === `openai/${id}`);
    if (exactOpenAI) return exactOpenAI.id;
    const bySuffix = models.filter((m) => m.id.endsWith(`/${id}`));
    if (bySuffix.length === 1) return bySuffix[0].id;
  }

  const needle = id.toLowerCase().split("/").pop();
  const similar = models
    .filter((m) => m.id.toLowerCase().includes(needle) || m.name?.toLowerCase().includes(needle))
    .map((m) => m.id);
  throw new Error(
    `"${id}" is not an image model on OpenRouter.` +
      (similar.length ? ` Image models matching it: ${similar.slice(0, 10).join(", ")}.` : "") +
      ` Call list_image_models for all ${models.length} image models.`
  );
}

/** Models tried in order for model "auto": IMAGE_DEFAULT_MODEL, then the newest gpt-image downwards. */
export async function defaultChain() {
  let ranked;
  try {
    ranked = rankGptImage(await imageModels()).map((m) => m.id);
  } catch {
    ranked = FALLBACK_CHAIN.map((id) => `openai/${id}`);
  }
  const pref = process.env.IMAGE_DEFAULT_MODEL?.trim();
  if (!pref) return ranked;
  const resolved = await resolveModel(pref);
  return [resolved, ...ranked.filter((id) => id !== resolved)];
}

async function modelInfo(id) {
  try {
    return (await imageModels()).find((m) => m.id === id);
  } catch {
    return undefined;
  }
}

/* ------------------------------------------------------- param translation */

const TIER_PX = { 512: 512, 768: 768, "1K": 1024, "1.5K": 1536, "2K": 2048, "4K": 4096 };

/** The enum value closest to `target`, measured by |log(value) - log(target)|. */
function nearest(values, target, valueOf) {
  let best;
  let bestDist = Infinity;
  for (const v of values) {
    const n = valueOf(v);
    if (!n) continue;
    const d = Math.abs(Math.log(n) - Math.log(target));
    if (d < bestDist) [best, bestDist] = [v, d];
  }
  return best;
}

const ratioOf = (s) => {
  const [w, h] = String(s).split(":").map(Number);
  return w > 0 && h > 0 ? w / h : undefined;
};

/**
 * Build the /images body for one model. Params the model does not list in
 * supported_parameters are dropped, and a WIDTHxHEIGHT size is translated to
 * aspect_ratio / resolution when the model has no `size`. Every such change is
 * reported back in `notes` so the caller sees what actually went out.
 */
function buildBody(model, info, params) {
  const sp = info?.supported_parameters;
  const body = { model, prompt: params.prompt };
  const notes = [];
  const supports = (k) => !sp || k in sp;

  const put = (k, v) => {
    if (v === undefined || v === null || v === "") return;
    if (!supports(k)) {
      notes.push(`${k}=${v} dropped (not supported by ${model})`);
      return;
    }
    const d = sp?.[k];
    if (d?.type === "enum" && !d.values.includes(v)) {
      notes.push(`${k}=${v} dropped (${model} accepts ${d.values.join("|")})`);
      return;
    }
    if (d?.type === "range" && (v < d.min || v > d.max)) {
      const clamped = Math.min(d.max, Math.max(d.min, v));
      notes.push(`${k}=${v} clamped to ${clamped}`);
      v = clamped;
    }
    body[k] = v;
  };

  if (params.n !== 1 || supports("n")) put("n", params.n);
  put("quality", params.quality);
  put("background", params.background);
  put("output_format", params.output_format);
  put("output_compression", params.output_compression);
  put("seed", params.seed);
  put("aspect_ratio", params.aspect_ratio);
  put("resolution", params.resolution);
  if (params.user) body.user = params.user;

  const size = params.size?.trim();
  if (size && size !== "auto") {
    const px = size.match(/^(\d+)x(\d+)$/i);
    const tier = Object.hasOwn(TIER_PX, size) ? size : null;
    if (supports("size")) {
      body.size = size;
    } else if (px) {
      const [w, h] = [Number(px[1]), Number(px[2])];
      const ar = sp?.aspect_ratio;
      if (ar && !body.aspect_ratio) {
        const pick = ar.type === "enum" ? nearest(ar.values, w / h, ratioOf) : `${w}:${h}`;
        if (pick) {
          body.aspect_ratio = pick;
          notes.push(`size ${size} sent as aspect_ratio ${pick}`);
        }
      }
      const res = sp?.resolution;
      if (res?.type === "enum" && !body.resolution) {
        const pick = nearest(res.values, Math.max(w, h), (v) => TIER_PX[v]);
        if (pick) {
          body.resolution = pick;
          notes.push(`size ${size} sent as resolution ${pick}`);
        }
      }
      if (!ar && !res) notes.push(`size ${size} dropped (${model} has no size control)`);
    } else if (tier && supports("resolution")) {
      put("resolution", tier);
    } else if (ratioOf(size) && supports("aspect_ratio")) {
      put("aspect_ratio", size);
    } else {
      notes.push(`size ${size} dropped (not understood by ${model})`);
    }
  }

  // OpenAI-only knobs: say so instead of failing silently.
  for (const k of ["moderation", "input_fidelity", "style"]) {
    if (params[k]) notes.push(`${k} ignored (OpenAI-only parameter)`);
  }
  return { body, notes };
}

const MEDIA_EXT = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/svg+xml": "svg" };

/** Shape the response like OpenAI's so the shared delivery code can save it. */
function normalize(res, notes) {
  const data = (res.data || []).map((img) => ({ ...img, output_format: MEDIA_EXT[img.media_type] }));
  return { ...res, data, notes };
}

export async function generate(model, params) {
  const { body, notes } = buildBody(model, await modelInfo(model), params);
  return normalize(await request("/images", { json: body }), notes);
}

const MIME = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
};

async function dataUrl(path) {
  const type = MIME[extname(path).toLowerCase()];
  if (!type) throw new Error(`Unsupported input image type "${extname(path) || path}". Use .png, .jpg, or .webp.`);
  return `data:${type};base64,${(await readFile(path)).toString("base64")}`;
}

export async function edit(model, params) {
  if (params.mask) {
    throw new Error(
      "OpenRouter's image API has no mask/inpainting. Describe the region to change in the " +
        "prompt instead, or switch to IMAGE_PROVIDER=openai for masked edits."
    );
  }
  const info = await modelInfo(model);
  const refs = info?.supported_parameters?.input_references;
  if (info && !refs) {
    const capable = rankGptImage((await imageModels()).filter((m) => m.supported_parameters?.input_references));
    throw new Error(
      `${model} cannot take input images. Use a model with input_references, e.g. ` +
        `${capable.slice(0, 3).map((m) => m.id).join(", ")} (list_image_models shows all).`
    );
  }
  if (refs?.max && params.images.length > refs.max) {
    throw new Error(`${model} takes at most ${refs.max} input image(s); got ${params.images.length}.`);
  }

  const { body, notes } = buildBody(model, info, params);
  body.input_references = await Promise.all(
    params.images.map(async (p) => ({ type: "image_url", image_url: { url: await dataUrl(p) } }))
  );
  return normalize(await request("/images", { json: body }), notes);
}
