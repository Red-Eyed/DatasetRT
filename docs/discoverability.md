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

> Python dataset caching for machine learning, powered by Rust.

Explain once that users install `dataset-rt` and import `dataset_rt`. Link GitHub,
PyPI, and documentation to each other. Share complete technical guides and
reproducible results where they answer a real question. Follow
[Google's SEO starter guide](https://developers.google.com/search/docs/fundamentals/seo-starter-guide)
for useful content, descriptive titles, and natural links.
