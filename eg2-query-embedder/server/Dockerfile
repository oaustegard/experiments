# EG2 query embedder for a Cloudflare Container: Node + onnxruntime-node (CPU) with the q4 model
# baked into the image, so a cold start never downloads from Hugging Face.
FROM node:22-slim

WORKDIR /app
COPY package.json package-lock.json ./
# Skip the CUDA download and keep only the linux/x64 onnxruntime binary (the package ships five
# platforms, ~290 MB; linux/x64 alone is 45 MB).
RUN ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm ci --omit=dev \
 && find node_modules/onnxruntime-node/bin -mindepth 3 -maxdepth 3 -type d ! -path '*/linux/x64' -exec rm -rf {} + \
 && npm cache clean --force

# 755, not 644: --chmod also applies to the model/ directory ADD creates, which needs +x to enter.
# Pinned to Hugging Face revision daa72c51243991dfcaf9f9137d2c573d8f7790c0; the checksums fail the build if the files change.
ADD --chmod=755 --checksum=sha256:f9eeba97acddf139b8ee2ddf04bc30dceafa88de93fadf74d7644e0d61a477a9 https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/daa72c51243991dfcaf9f9137d2c573d8f7790c0/onnx/model_q4.onnx model/model_q4.onnx
ADD --chmod=755 --checksum=sha256:c3975f2d1ab7a1878ae31a7d7a9b7804a827aff3800b60dfceafce21cac3df49 https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/daa72c51243991dfcaf9f9137d2c573d8f7790c0/onnx/model_q4.onnx_data model/model_q4.onnx_data
ADD --chmod=755 --checksum=sha256:4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4 https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/daa72c51243991dfcaf9f9137d2c573d8f7790c0/tokenizer.json model/tokenizer.json
ADD --chmod=755 --checksum=sha256:17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874 https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/daa72c51243991dfcaf9f9137d2c573d8f7790c0/tokenizer_config.json model/tokenizer_config.json
ADD --chmod=755 --checksum=sha256:031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5 https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/daa72c51243991dfcaf9f9137d2c573d8f7790c0/config_sentence_transformers.json model/config_sentence_transformers.json

COPY server.cjs test_parity.mjs ./
# The card-score check runs at build time as the runtime user, so an image that embeds wrongly,
# or cannot read its own model, never ships.
USER node
RUN node test_parity.mjs

ENV PORT=8080 NODE_ENV=production
EXPOSE 8080
CMD ["node", "server.cjs"]
