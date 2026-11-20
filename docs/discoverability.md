---
title: Documentation Publishing and Google Search Visibility
description: Build and publish DatasetRT documentation, verify search indexing, and measure branded and problem-based discovery in Google Search Console.
---

# Documentation publishing and Google search visibility

## Build and publish

`just docs-build` synchronizes the public API documentation from docstrings and
builds HTML into the ignored `site/` directory. It fails on documentation warnings,
including missing internal pages and anchors. `just docs-serve` performs the same
synchronization before opening a local development server.

Pull requests build documentation without deployment permissions. Pushes to
`main` build and deploy through `.github/workflows/docs.yml` to
<https://red-eyed.github.io/DatasetRT/>. The workflow installs only documentation
dependencies and does not build a native wheel. Python package artifacts remain
locally built and published only by the maintainer.

`mkdocs.yml` defines the canonical base URL and navigation. MkDocs emits HTML,
canonical links, descriptions, a local search index, and `sitemap.xml`. Give new
pages a descriptive first heading and a `description` in YAML front matter. Add
them to navigation and link them from a relevant guide. Keep source-code links
absolute to GitHub so they also work on the hosted documentation.

The public build explicitly excludes investigation notes, inversion reviews,
the detailed internal runtime/architecture records, build/release instructions,
development instructions, and this maintainer page. Keep them out of navigation
and user-guide links. `runtime-guide.md` and `overview.md` explain the application
contracts separately. Removing a page from navigation alone does not prevent
MkDocs from publishing or indexing it; preserve the `exclude_docs` entries.

## Verify Google indexing

After the public site responds successfully:

1. Add the URL-prefix property `https://red-eyed.github.io/DatasetRT/` in
   [Google Search Console](https://search.google.com/search-console/about).
2. Complete Google's ownership verification. The required verification token
   comes from the maintainer's Google account; it cannot be inferred by a build.
3. Submit `https://red-eyed.github.io/DatasetRT/sitemap.xml`.
4. Inspect the homepage and tutorial URLs with URL Inspection. Check the rendered
   page, selected canonical URL, indexing eligibility, and any reported failures.
5. Request indexing where appropriate, then review the indexing report after
   Google has had time to crawl the pages.

A `site:red-eyed.github.io/DatasetRT/` search is a useful spot check, but it is
not a complete indexing inventory. Search Console provides the owner-facing
diagnostics. Follow [Google's developer guidance](https://developers.google.com/search/docs/fundamentals/get-started-developers).

## Measure discovery

Record a baseline once Search Console has data. Compare the following over
successive 28-day windows, allowing for crawl delays and small initial volumes:

| Audience | Example queries | Measures |
| --- | --- | --- |
| People who know the name | `DatasetRT`, `dataset-rt`, `dataset_rt` | Impressions, clicks, and destination pages |
| People solving a problem | `Python dataset cache`, `cache preprocessed training data`, `deterministic weighted sampling` | Impressions, clicks, click-through rate, and query/page matches |
| People evaluating adoption | Queries mentioning PyTorch, Polars, LMDB, or WebDataset | Tutorial/comparison visits and relevant issues or feedback |

These phrases are content hypotheses, not measured keyword demand. Use observed
queries to improve explanations and examples. Search position varies by query,
location, and time; publishing a sitemap does not guarantee indexing or ranking.

## Keep the project recognizable

Use **DatasetRT** as the display name and the same short description in the
README, documentation, GitHub About, and package summaries:

> Rust-powered dataset cache for Python with separate Polars metadata for analysis,
> filtering, and weighted sampling.

Lead introductions with the outcome:

> Cache preprocessing once. Change subsets and sampling weights without rewriting payloads.

Explain once that users install `dataset-rt` and import `dataset_rt`. Link GitHub,
PyPI, and documentation to each other. Share complete technical guides and
reproducible results where they answer a real question. Follow
[Google's SEO starter guide](https://developers.google.com/search/docs/fundamentals/seo-starter-guide)
for useful content, descriptive titles, and natural links.

## Repository presentation

The README links installation, a complete example, tutorials, and the existing
benchmark evidence. `CITATION.cff` provides software citation metadata;
`CONTRIBUTING.md` and the GitHub templates explain how to report and validate
changes. The [social-preview instructions](assets/social-preview.md) include the
upload-ready artwork and generation prompt.

Prepared GitHub About settings (apply separately from repository commits):

- Description: `Rust-powered dataset cache for Python. Separate Polars metadata for analysis, filtering, and weighted sampling.`
- Website: `https://red-eyed.github.io/DatasetRT/`
- Topics: retain `data-loading`, `dataset`, `dataset-cache`, `machine-learning`,
  `polars`, `python`, `reproducibility`, and `rust`; add `pytorch` for the documented
  optional integration.

## Announcement drafts

These are drafts for maintainer review. Nothing here posts to an external
community. Choose a venue where the workflow is relevant and check its current
posting rules before sharing.

### Short introduction

I built DatasetRT for ML experiments that reuse preprocessed training records
but change subsets or sampling weights. It keeps serialized payloads in immutable
disk shards and exposes separate metadata through Polars, so you can filter a
development subset or rebalance classes without rewriting cached records.
Rust owns indexed reads, deterministic sampling, and bounded prefetching;
Python owns serialization and transforms. PyTorch integration is optional.

Try it: `pip install dataset-rt` or `uv add dataset-rt`.
[Runnable quickstart](https://red-eyed.github.io/DatasetRT/cache-training-data/) ·
[Repository](https://github.com/Red-Eyed/DatasetRT)

### Technical introduction

DatasetRT: cache preprocessing once, then change how you sample with Polars.

For workflows that repeatedly train from the same preprocessed records,
DatasetRT separates immutable payload storage from the active metadata table.
Filter rows, duplicate entries, or adjust weights without decoding the payloads
or rebuilding the cache. Rust validates sample identities and owns the read/write
pipelines; Python supplies serialized records and training transforms.

Shuffled reads use weighted sampling with replacement. Source names control
cache reuse, so preprocessing changes need a new source name or cache root.
Files must be available through the filesystem; this is not a remote streaming
service.

The docs include [Polars sampling examples](https://red-eyed.github.io/DatasetRT/polars-sampling/),
a [PyTorch tutorial](https://red-eyed.github.io/DatasetRT/pytorch-dataloader/),
and [reproducible reader measurements](https://red-eyed.github.io/DatasetRT/benchmarks/).
Those measurements compare DatasetRT reader configurations on a synthetic local
workload; they do not establish a speed ranking against other libraries.

I'd welcome feedback from people who change training populations frequently:
what would you need to evaluate this in your pipeline?

[Source and installation](https://github.com/Red-Eyed/DatasetRT)
