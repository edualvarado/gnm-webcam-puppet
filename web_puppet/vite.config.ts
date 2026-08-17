import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vite';

const here = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  // Relative so a built page works from a file path or any subdirectory,
  // not just a domain root.
  base: './',
  server: { port: 5173 },
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        // Two independent pages: the viewer, and the tracker. They share
        // nothing at runtime on purpose -- M3 has to be judgeable without the
        // head in the way.
        main: resolve(here, 'index.html'),
        tracker: resolve(here, 'tracker.html'),
      },
    },
    // The 17 MB model lives in public/ and is copied verbatim; warning about
    // it on every build is noise.
    chunkSizeWarningLimit: 2048,
  },
});
