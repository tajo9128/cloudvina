# BioDockify Learning Portal

**learn.biodockify.com** - Comprehensive educational platform for molecular docking and drug discovery.

## 🎓 What's Inside

A complete learning experience featuring:
- **13 interactive lessons** across 5 modules
- **Progress tracking** - Mark lessons complete
- **Responsive design** - Works on all devices
- **Rich content** - Code examples, tutorials, best practices

## 📚 Course Modules

### 1. Getting Started
- Welcome to BioDockify
- Creating Your Account
- Your First Docking Job

### 2. Fundamentals
- What is Molecular Docking?
- Understanding File Formats

### 3. Running Jobs
- Preparing Receptor Files
- Preparing Ligand Files
- Docking Parameters

### 4. Advanced Topics
- Interpreting Results
- Virtual Screening

### 5. API & Integration
- BioDockify API Overview

## 🚀 Quick Start

### Install Dependencies
```bash
npm install
```

### Start Development Server
```bash
npm run dev
# Opens at http://localhost:5173
```

### Build for Production
```bash
npm run build
# Output in dist/
```

### Preview Production Build
```bash
npm run preview
```

## 🛠 Tech Stack

- **React 18.3.1** - UI framework
- **Vite 6.0.5** - Build tool
- **TailwindCSS 3.4.17** - Styling
- **React Router** - Navigation
- **React Markdown** - Lesson content
- **Lucide React** - Icons

## 📁 Project Structure

```
web_learn/
├── src/
│   ├── App.jsx       # Main application with all lessons
│   ├── main.jsx      # React entry point
│   └── index.css     # Tailwind imports
├── index.html        # Entry HTML
├── package.json      # Dependencies
├── vite.config.js    # Vite config
├── tailwind.config.js # Tailwind theme
└── vercel.json       # Deployment config
```

## 🎨 Features

- ✅ **Dark Mode Theme** - Easy on the eyes
- ✅ **Progress Tracking** - LocalStorage-based
- ✅ **Sidebar Navigation** - Collapsible on mobile
- ✅ **Code Highlighting** - For examples and snippets
- ✅ **Responsive Design** - Desktop, tablet, mobile
- ✅ **SEO Optimized** - Meta tags included

## 🌐 Deployment

This project is configured for deployment on **Vercel**.

### Deploy to Vercel:
1. Push to GitHub
2. Import project in Vercel
3. Set root directory to `web_learn`
4. Deploy!

Or use Vercel CLI:
```bash
vercel --prod
```

### Custom Domain
Add `learn.biodockify.com` in Vercel dashboard:
- Settings → Domains → Add `learn.biodockify.com`
- Configure DNS CNAME: `learn` → `cname.vercel-dns.com`

## 📝 Adding New Lessons

Edit `src/App.jsx` and add to the `lessons` array:

```javascript
{
  id: 'advanced-screening',
  title: 'Advanced Screening Techniques',
  description: 'Master virtual screening workflows',
  content: (
    <>
      <h2>Your Lesson Title</h2>
      <p>Your content here...</p>
    </>
  )
}
```

## 🔧 Configuration

### Environment Variables
Create `.env` file (optional):
```
VITE_API_URL=https://api.biodockify.com
VITE_ANALYTICS_ID=your-analytics-id
```

### Tailwind Theme
Customize colors in `tailwind.config.js`:
```javascript
colors: {
  primary: {
    500: '#0ea5e9',  // Customize your brand color
  }
}
```

## 📊 Build Stats

- **Bundle size**: 197 kB (59 kB gzipped)
- **CSS**: 14 kB (3.3 kB gzipped)
- **Build time**: ~2.5 seconds
- **Dependencies**: 269 packages

## 🤝 Contributing

To add new content:
1. Edit lessons in `src/App.jsx`
2. Test locally with `npm run dev`
3. Build with `npm run build`
4. Commit and push to GitHub
5. Vercel auto-deploys!

## 📄 License

Part of the BioDockify platform. See main repository for license details.

## 🔗 Links

- **Production**: https://learn.biodockify.com
- **Main Platform**: https://www.biodockify.com
- **AI Suite**: https://ai.biodockify.com
- **API Docs**: https://docs.biodockify.com

---

Built with ❤️ for making molecular docking education accessible to everyone.
