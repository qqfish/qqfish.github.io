# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This is a Jekyll blog site using GitHub Pages. Key commands:

- **Local development**: `bundle exec jekyll serve` (starts local server at http://localhost:4000)
- **Install dependencies**: `bundle install` (installs Ruby gems from Gemfile)
- **Build site**: `bundle exec jekyll build` (generates static site in _site directory)

## Repository Structure

This is a Jekyll-based blog repository with the following architecture:

- **Content files**: Markdown files in root (`index.md`, `about.md`) and `_posts/` directory
- **Configuration**: `_config.yml` controls site settings, theme (minima), and plugins
- **Layouts & includes**: `_includes/` directory contains reusable HTML components (alert.html, info.html, etc.)
- **Styling**: `assets/main.scss` imports theme styles and adds custom CSS for posts, images, and Jupyter notebook formatting
- **Dependencies**: `Gemfile` manages Ruby gems including Jekyll, GitHub Pages, and plugins

## Content Management

- Blog posts must follow naming convention: `YEAR-MONTH-DAY-filename.md` in `_posts/` directory
- Posts support front matter, Markdown formatting, and custom includes for alerts/info boxes
- Images should be placed in `images/` directory
- Custom SCSS styling optimized for Python code blocks and Jupyter notebook output

## Jekyll Configuration

- Uses minima theme with GitHub Pages deployment
- Plugins: jekyll-feed, jekyll-gist, jekyll-octicons, jekyll-github-metadata
- Kramdown markdown processor with KaTeX math engine support
- Rouge syntax highlighter with custom styling for Python code