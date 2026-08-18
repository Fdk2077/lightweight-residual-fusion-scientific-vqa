# Security and privacy check

Before publication, this repository was screened for:

- API keys and credential-like strings
- local absolute filesystem paths
- raw model checkpoints
- raw benchmark images
- row-level temporal PMC-VQA question/answer audit exports

No credentials are intentionally included. The temporal audit supports an optional NCBI API key via an environment variable; no key value is stored here.

## Licensing decision

- Custom repository code: MIT License
- Associated article: intended CC BY 4.0 at Communications Medicine publication stage
- Third-party data/models/images: original provider licenses apply
