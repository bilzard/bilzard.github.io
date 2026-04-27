## Bilzard's Blog

This repository contains the source code for my personal blog hosted on GitHub Pages.

- **Website**: [https://bilzard.github.io](https://bilzard.github.io)

## Development

### 1. Setup
Ensure you have Ruby and Bundler installed. Then, install dependencies:

```bash
bundle install
```

### 2. Local Preview
To start the Jekyll server and preview your blog locally:

```bash
bundle exec jekyll serve --livereload --unpublished
```
- Access via: `http://localhost:4000`

### 3. Create a New Post
Use the Rake task to generate a new post file with the correct front matter and filename format.

```bash
bundle exec rake post TITLE="Your Post Title"
```

- **File location**: `_posts/YYYY-MM-DD-your-post-title.md`
- **Default Front Matter**:
  - `latex: true` (MathJax/KaTeX enabled)
  - `toc: true` (Table of Contents enabled)
  - `categories: blog`

## License

The articles, images, and other non-code assets in this repository are fully protected under copyright law.

Copyright © 2024 Bilzard. All Rights Reserved.

## External Resources

This project uses the following external resources:

1. [jekyll](https://github.com/jekyll/jekyll) - MIT license
2. [jekyll-toc](https://github.com/allejo/jekyll-toc) - MIT license