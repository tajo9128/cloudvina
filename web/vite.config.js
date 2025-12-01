import { defineConfig } from 'vite'
// Force Vercel Rebuild: Cache Buster v2.1.1
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        port: 3000
    },
    base: '/',
    build: {
        outDir: 'dist'
    }
})
