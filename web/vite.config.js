import { defineConfig } from 'vite'
// Force Vercel Rebuild: Cache Buster v3.1.0
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        port: 3000
    },
    base: '/',
    build: {
        outDir: 'dist'
    },
    resolve: {
        preserveSymlinks: true
    }
})
