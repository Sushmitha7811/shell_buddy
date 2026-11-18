# 6. Hugging Face Spaces (tried, not possible on the free plan)

**Status: not built.** This page records what I tried and why it didn't work, so you don't have to find out the same way.

## The idea

After AWS used up free credits, the plan was to move the [AWS setup](../4-aws-ec2/) to a **Hugging Face Space**: a small server hosted by Hugging Face. Users would only need `howto.py` and the Space's URL (`https://<user>-shell-buddy.hf.space`), with no Ollama install and no model download.

```
howto.py ──HTTPS──► Space (Docker): gateway.py → Ollama → model downloaded from the Hub
```

The plan:
1. A **Docker** Space running Ollama and `gateway.py` on port 7860
2. At startup, download the model from the [Hub repo](../5-huggingface-hub/)
3. API keys stored in the Space's **Secrets** settings instead of `/etc/shell-buddy.env`

A 0.5B model fits easily on the free "CPU basic" hardware that Hugging Face documents (2 CPUs, 16 GB RAM).

## What actually happened (November 2026)

On the **Create new Space** page of a free account:
- **Static** was the only free SDK. **Gradio and Docker were marked as paid.**
- There was **no hardware option** to choose at all.

The official docs still describe free CPU basic Spaces (they sleep after 48 hours without use). But forum posts from mid-2026 report that the free tier was quietly restricted, which matches what I saw.

## Why a Static Space doesn't help

A Static Space only serves files (HTML, JavaScript, images) to a web browser. It can't run a program in the background, so there's nothing for `howto` to call from a terminal.

A static page *could* run the model inside the visitor's browser with WebAssembly (for example [wllama](https://github.com/ngxson/wllama), which loads GGUF files). That would be a web page rather than a terminal tool, so it's a different project.

## What I did instead

[Approach 5](../5-huggingface-hub/): publish the model on the Hub and let each user run it locally with Ollama. It's free for everyone and faster than a shared CPU server. The cost is a one-time 531 MB download per user.

## If you have a paid Hugging Face plan

The plan above should work as described. Reuse `gateway.py` from [approach 4](../4-aws-ec2/). Change `MODEL` to `hf.co/SushmithaJujare/shell-buddy-GGUF`, or `ollama cp` the model to `shell-buddy`, and listen on port 7860. Keep the API keys: a public Space URL can be found by anyone.

## Sources

- [Hub docs: Spaces hardware](https://huggingface.co/docs/hub/main/spaces-gpus)
- [Hub docs: Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview)
- [Forum: community complaint about the free CPU Basic change](https://discuss.huggingface.co/t/official-community-complaint-revert-free-cpu-basic-spaces-and-remove-anti-developer-sdk-restrictions/177703)
- [Forum: CPU Basic quota limit with only one Space](https://discuss.huggingface.co/t/cpu-basic-quota-limit-with-only-one-space/180646)
