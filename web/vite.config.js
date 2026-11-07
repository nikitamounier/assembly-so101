import { defineConfig } from "vite";

export default defineConfig({
  build: {
    outDir: "../assembly_policy/viewer/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 700,
  },
});
