import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [tailwindcss(), react()],
  optimizeDeps: {
    include: ['plotly.js-dist-min'],
  },
  server: {
    port: 5173,
    proxy: {
      '/sessions': 'http://localhost:8000',
      '/datasets': 'http://localhost:8000',
      '/clusters': 'http://localhost:8000',
      '/turns':    'http://localhost:8000',
      '/umap':     'http://localhost:8000',
    },
  },
})
