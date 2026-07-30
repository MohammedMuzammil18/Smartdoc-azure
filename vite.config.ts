import { defineConfig } from 'vite'
import { fileURLToPath, URL } from 'node:url'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'


function figmaAssetResolver() {
  return {
    name: 'figma-asset-resolver',
    resolveId(id: string) {
      if (id.startsWith('figma:asset/')) {
        const filename = id.replace('figma:asset/', '')
        return fileURLToPath(new URL('src/assets/' + filename, import.meta.url))
      }
    },
  }
}

export default defineConfig({
  plugins: [
    figmaAssetResolver(),
    // The React and Tailwind plugins are both required for Make, even if
    // Tailwind is not being actively used – do not remove them
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },

  server: {
    proxy: {
      '/predict': 'http://127.0.0.1:5000',
      '/search': 'http://127.0.0.1:5000',
      '/add-doc': 'http://127.0.0.1:5000',
      '/update-doc': 'http://127.0.0.1:5000',
      '/delete-doc': 'http://127.0.0.1:5000',
      '/register': 'http://127.0.0.1:5000',
      '/login': 'http://127.0.0.1:5000',
      '/admin': 'http://127.0.0.1:5000',
      '/health': 'http://127.0.0.1:5000',
      '/autocomplete': 'http://127.0.0.1:5000',
      '/azure': 'http://127.0.0.1:5000',
      '/search-history': 'http://127.0.0.1:5000',
    }
  },
  // File types to support raw imports. Never add .css, .tsx, or .ts files to this.
  assetsInclude: ['**/*.svg', '**/*.csv'],
})
