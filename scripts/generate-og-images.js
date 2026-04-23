const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

(async () => {
  const jsonPath = path.join(__dirname, '../_site/assets/og-data.json');
  const posts = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  const outputDir = path.join(__dirname, '../_site/assets/images/og');

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    headless: "new",
    args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage' // メモリ不足エラー防止のため追加を推奨
    ]
});
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 630 });

  for (const post of posts) {
    console.log(`Generating: ${post.filename}`);

    // カードのデザイン（HTML/CSS）
    const html = `
      <html>
        <head>
          <style>
            body {
              width: 1200px; height: 630px; margin: 0;
              display: flex; flex-direction: column; align-items: center; justify-content: center;
              background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); /* variables.scss由来のSlate色 */
              font-family: 'Open Sans', sans-serif;
              color: #0f172a; border: 20px solid #ffffff; box-sizing: border-box;
            }
            .emoji { font-size: 100px; margin-bottom: 20px; }
            .title { font-size: 60px; font-weight: 800; text-align: center; padding: 0 80px; line-height: 1.4; }
            .site-name { position: absolute; bottom: 60px; font-size: 24px; color: #64748b; font-weight: bold; }
          </style>
        </head>
        <body>
          <div class="emoji">${post.emoji}</div>
          <div class="title">${post.title}</div>
          <div class="site-name">bilzard.make</div>
        </body>
      </html>
    `;

    await page.setContent(html);
    await page.screenshot({ path: path.join(outputDir, post.filename) });
  }

  await browser.close();
})();