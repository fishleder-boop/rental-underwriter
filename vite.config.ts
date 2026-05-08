import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5300,
    allowedHosts: ['.ngrok-free.dev', '.ngrok.app', '.ngrok.io', '.trycloudflare.com'],
  },
})
