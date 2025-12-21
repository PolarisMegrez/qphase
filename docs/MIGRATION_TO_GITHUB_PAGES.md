---
layout: default
title: Migration Guide
---
# Migration from MkDocs to GitHub Pages

This document describes the migration from MkDocs to GitHub Pages with Jekyll.

## Changes Made

### 1. Jekyll Configuration
- Created `_config.yml` with Jekyll settings
- Configured Kramdown with MathJax support
- Set up minima theme (compatible with GitHub Pages)
- Configured proper exclusions for build

### 2. Directory Structure
```
docs/
├── _config.yml          # Jekyll configuration
├── _layouts/            # Jekyll layouts
│   └── default.html     # Default layout with MathJax
├── _includes/           # Includes directory (empty, for future use)
├── Gemfile              # Ruby dependencies
├── Gemfile.lock         # Locked dependency versions
├── index.md             # Homepage
├── user_guide/          # User documentation
├── dev_guide/           # Developer documentation
├── api/                 # API documentation
└── ecosystem/           # Plugin ecosystem documentation
```

### 3. Front Matter
Added YAML front matter to all markdown files:
```yaml
---
layout: default
title: Page Title
---
```

### 4. MathJax Support
- Configured in `_config.yml` with `kramdown: math_engine: mathjax`
- Added MathJax script to `_layouts/default.html`
- Supports both inline `$...$` and display `$$...$$` math

### 5. GitHub Actions Workflow
Updated `.github/workflows/docs.yml` to:
- Use `ruby/setup-ruby` for Jekyll build
- Build with `bundle exec jekyll build`
- Deploy to GitHub Pages using `actions/deploy-pages`
- Proper caching for faster builds

## Next Steps

### 1. Enable GitHub Pages
1. Go to repository Settings → Pages
2. Select "GitHub Actions" as the source
3. The workflow will automatically run on push to main

### 2. Remove MkDocs
After successful migration, remove:
- `mkdocs.yml` (no longer needed)
- Python dependencies for MkDocs in requirements.txt
- MkDocs-related CI cache keys

### 3. Verify Deployment
1. Push changes to main branch
2. Check Actions tab for successful build
3. Visit `https://<username>.github.io/<repo>/` to view site

## Theme Customization

The current theme is `jekyll/minima`. To customize:

1. Create `_includes/` files to override theme components
2. Add custom CSS to the layout
3. Or switch to another GitHub Pages compatible theme

## Math in Documents

Use standard LaTeX syntax:
- Inline: `$E = mc^2$`
- Display: `$$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$`

## Local Development

To preview changes locally:

```bash
cd docs
bundle install
bundle exec jekyll serve
```

Visit `http://localhost:4000` to view the site.

## Troubleshooting

### Build Fails
- Check Ruby version (requires 3.0+)
- Ensure Gemfile.lock is committed
- Verify all dependencies in Gemfile

### Math Not Rendering
- Check that MathJax script is loaded
- Verify `math_engine: mathjax` in _config.yml
- Ensure front matter has `layout: default`

### Links Not Working
- Use relative paths from docs root
- Jekyll uses permalink: pretty (no .html extension)
- Navigation links updated in default.html
