// src/index.js
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import theme from "./Theme";

import { ThemeProvider } from "@mui/material/styles";
import { CssBaseline } from "@mui/material";

import { CacheProvider } from "@emotion/react";
import createCache from "@emotion/cache";
import stylisRTLPlugin from "stylis-plugin-rtl";

const cacheRtl = createCache({
  key: "muirtl",
  stylisPlugins: [stylisRTLPlugin],
});

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <CacheProvider value={cacheRtl}>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </CacheProvider>
);
