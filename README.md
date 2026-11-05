# Latest AI

Watches Hugging Face and news feeds for image and video generation and editing
releases, then builds one page: `data/digest.html`. It needs Python 3.10+ and
nothing else.

```bash
./latestai.py --open                     # check now and open the digest
./latestai.py install-timer              # check every 6 hours, with desktop notifications
./latestai.py install-timer --every-hours 12
./latestai.py uninstall-timer
python3 -m unittest                      # run the tests
```

## What gets shown

Hugging Face gets dozens of new uploads a day, and most are LoRAs and reuploads.
A model from the last 30 days appears only if it passes these rules:

| Model | Shown when |
|---|---|
| Original model from a watched org | always |
| LoRA, quantized model or finetune from a watched org | trending 10+ or 50+ likes |
| Original model from anyone else | trending 15+ or 50+ likes |
| LoRA, quantized model or finetune from anyone else | trending 50+ or 200+ likes |

Models tagged or named as NSFW are hidden. Sections follow the Hugging Face
pipeline tag: `text-to-image` is image generation, `image-to-image` is image
editing, `text-to-video` and `image-to-video` are video generation, and
`video-to-video` is video editing. Uploaders choose these tags, so some models
land in an odd section.

Within each section, models are ranked by Hugging Face trending score, with likes
breaking ties. Papers and news have no such score, so the newest come first.

News feeds and papers are filtered by keyword, using the `media` and `release`
patterns in `config.json`. When a new product name appears, add it to `media` or
those stories may be missed.

## Trying a model

Each model on the page shows how to try it, checked again on every run:

| What you see | What it needs |
|---|---|
| Try demo, "No signup needed" | Nothing. ZeroGPU demos allow about 2 minutes of GPU time a day, or 5 with a free account |
| Try demo, "May ask for a free Hugging Face login" | The demo has sign-in built in, and it may or may not be required |
| Hosted API | A Hugging Face account and paid credits |
| No live demo found | Download the files and run them yourself |

A demo is offered only if its Space is running or asleep and is either owned by the
model's publisher or shares a distinctive word with the model's name. Community demos
can be run by anyone, so don't upload anything private to them. Quotas come from the
[ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu) and may change.

## Changing it

Everything is in `config.json`: watched orgs, thresholds, feeds and keywords. To
add a feed, add its name and URL. Leave out `filter` to keep every post.

If a source fails, the digest lists it at the top and the other sources still run.
Items seen so far are stored in `data/state.db`. Delete `data/` to start over.
