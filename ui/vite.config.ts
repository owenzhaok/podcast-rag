import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  test: {
    globals: true,
    environment: 'jsdom',
  },
  server: {
    proxy: {
      '/search': 'http://localhost:8000',
      '/ask': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
