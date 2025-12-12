// Script to parse all 20 blog markdown files and extract metadata

import fs from 'fs';
import path from 'path';

const blogDir = path.join(process.cwd(), '../web/src/pages/blog');
const files = fs.readdirSync(blogDir).filter(f => f.endsWith('.md'));

console.log(`Found ${files.length} markdown files`);

const blogPosts = files.map((filename, index) => {
  const content = fs.readFileSync(path.join(blogDir, filename), 'utf-8');
  
  // Extract frontmatter
  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---/);
  const metadata = {};
  
  if (frontmatterMatch) {
    const frontmatter = frontmatterMatch[1];
    frontmatter.split('\n').forEach(line => {
      const [key, ...valueParts] = line.split(':');
      if (key && valueParts.length) {
        let value = valueParts.join(':').trim();
        // Remove quotes
        value = value.replace(/^["']|["']$/g, '');
        // Parse arrays
        if (value.startsWith('[')) {
          value = JSON.parse(value.replace(/'/g, '"'));
        }
        metadata[key.trim()] = value;
      }
    });
  }
  
  // Extract first paragraph as excerpt
  const contentBody = content.replace(/^---[\s\S]*?---/, '').trim();
  const firstPara = contentBody.split('\n\n')[1] || contentBody.split('\n\n')[0];
  const excerpt = firstPara?.replace(/^#+\s+/, '').replace(/\*\*/g, '').substring(0, 200) + '...';
  
  return {
    id: index + 1,
    title: metadata.title,
    slug: filename.replace('.md', ''),
    excerpt: excerpt || metadata.description,
    content: content body,
    author: metadata.author,
    date: metadata.date,
    category: metadata.category,
    readTime: metadata.readTime,
    tags: metadata.keywords || [],
    image: `https://images.unsplash.com/photo-${Math.floor(Math.random() * 1000000000)}?w=1200&auto=format&fit=crop`,
    featured: index < 4 // Mark first 4 as featured
  };
});

console.log(JSON.stringify(blogPosts, null, 2));
