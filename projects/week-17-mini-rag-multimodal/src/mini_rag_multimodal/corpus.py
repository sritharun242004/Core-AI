"""A tiny, deterministic corpus used by every offline example."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    links: tuple[str, ...] = ()


DOCUMENTS = (
    Document(
        "clip",
        "CLIP aligns images and language",
        "Contrastive image text pretraining aligns visual and language representations. "
        "A shared embedding makes zero shot image retrieval possible.",
        ("vit", "rag"),
    ),
    Document(
        "vit",
        "ViT turns image patches into tokens",
        "A vision transformer splits an image into patches, adds positional information, "
        "and applies self attention to visual tokens.",
        ("clip", "vae"),
    ),
    Document(
        "vae",
        "VAEs learn a compact latent space",
        "A variational autoencoder encodes pixels into a stochastic latent and decodes a "
        "reconstruction. The KL term regularizes the latent distribution.",
        ("gan", "diffusion"),
    ),
    Document(
        "gan",
        "GANs learn by a generator and discriminator game",
        "A generative adversarial network trains a generator against a discriminator. "
        "The generator makes samples while the discriminator distinguishes real data.",
        ("vae", "diffusion"),
    ),
    Document(
        "diffusion",
        "Diffusion models denoise a noisy image",
        "A diffusion model adds noise and learns a denoising process that reverses it. "
        "Repeated denoising steps generate an image from a noisy sample.",
        ("vae", "video"),
    ),
    Document(
        "video",
        "World and video models predict temporal state",
        "A world model predicts how a scene changes over time. Video generation must model "
        "spatial structure, motion, and temporal consistency.",
        ("diffusion", "voice"),
    ),
    Document(
        "voice",
        "Voice pipelines handle spoken audio",
        "Speech recognition transcribes spoken audio, a language model plans a response, "
        "and speech synthesis renders an audible answer. Streaming can reduce latency.",
        ("video", "rag"),
    ),
    Document(
        "rag",
        "RAG grounds generation in retrieved evidence",
        "Retrieval augmented generation searches an indexed corpus before generation. "
        "The context window should preserve citations and evidence boundaries.",
        ("clip", "voice"),
    ),
)

QUERIES = {
    "q_diffusion": "diffusion noise denoising",
    "q_voice": "spoken audio pipeline",
    "q_retrieval": "retrieval evidence context",
    "q_visual": "image patches language embeddings",
}

# Relevance is graded: 2 is a particularly useful answer, 1 is related context.
QRELS = {
    "q_diffusion": {"diffusion": 2, "vae": 1},
    "q_voice": {"voice": 2, "video": 1},
    "q_retrieval": {"rag": 2, "clip": 1},
    "q_visual": {"clip": 2, "vit": 2, "vae": 1},
}

# Controlled aliases are not a pretrained embedding model. They are a transparent
# teaching fixture that lets a dense baseline demonstrate semantic-ish matches.
ALIASES = {
    "utterances": "speech",
    "spoken": "speech",
    "speech": "speech",
    "audio": "speech",
    "voice": "speech",
    "transcribe": "speech",
    "transcription": "speech",
    "images": "image",
    "pixels": "image",
    "visual": "image",
    "patches": "patch",
    "denoise": "denoising",
    "noisy": "noise",
    "generates": "generate",
    "generation": "generate",
    "retrieved": "retrieve",
    "retrieval": "retrieve",
}
