# Govinda M. Kamath — personal website

Personal academic website built with [al-folio](https://github.com/alshedivat/al-folio) (v1.1), hosted on GitHub Pages.

## Structure

| Path | Content |
| --- | --- |
| `_pages/about.md` | Homepage (bio, profile photo, selected papers, news, recent posts) |
| `_pages/research.md` | Research interests and past work |
| `_bibliography/papers.bib` | Publications — edit this and the publications page updates automatically |
| `_pages/publications.md` | Auto-generated from `papers.bib` |
| `_data/cv.yml` | HTML CV data (also links to PDF CVs in `assets/pdf/`) |
| `_data/socials.yml` | Social links |
| `_data/coauthors.yml` | Coauthor profile links |
| `_data/venues.yml` | Venue abbreviations, links, and colors |
| `_news/` | News items (shown on the homepage) |
| `_projects/` | Project pages |
| `_posts/` | Blog posts |
| `_teachings/` | Courses taught |
| `assets/pdf/` | `GK_CV.pdf` and `gk_Resume.pdf` |
| `assets/img/prof_pic.jpg` | Profile photo |

## Local development

```bash
export PATH="/usr/local/opt/ruby/bin:/usr/bin:/bin:$PATH"   # brew ruby, Apple clang/ld
bundle install
bundle exec jekyll serve
```

Then open <http://localhost:4000>.

> Note: if gem native extensions fail to build, make sure `ld` and `clang` resolve to
> Apple's tools (`/usr/bin/`) and not to a Homebrew/anaconda toolchain.

## Deployment

Pushing to `master` triggers `.github/workflows/deploy.yml`, which builds the site and
publishes it to the `gh-pages` branch. In the repository settings, GitHub Pages must be
configured to serve from the `gh-pages` branch. See `docs/INSTALL.md` for details.
