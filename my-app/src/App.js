// src/App.js
import React, { useState } from "react";
import { Container, Typography, TextField, Button, Box } from "@mui/material";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      setResult(data.result);
    } catch (err) {
      setResult("שגיאה בשליחת הבקשה לשרת");
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 10, textAlign: "right" }}>
      <Typography variant="h4" gutterBottom>
        הכנס טקסט לעיבוד
      </Typography>
      <Box component="form" onSubmit={handleSubmit} noValidate>
        <TextField
          fullWidth
          label="טקסט"
          variant="outlined"
          value={text}
          onChange={(e) => setText(e.target.value)}
          inputProps={{ dir: "rtl", style: { textAlign: "right" } }}
          sx={{ mb: 2 }}
        />
        <Button type="submit" variant="contained" fullWidth>
          שלח
        </Button>
      </Box>
      {result && (
        <Typography variant="h6" sx={{ mt: 4 }}>
          תוצאה: {result}
        </Typography>
      )}
    </Container>
  );
}

export default App;
