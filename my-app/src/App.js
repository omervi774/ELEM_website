// src/App.js
import React, { useState } from "react";
import {
  Container,
  Typography,
  Button,
  Box,
  Input,
  Alert,
} from "@mui/material";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile && uploadedFile.name.endsWith(".csv")) {
      setFile(uploadedFile);
      setError("");
    } else {
      setFile(null);
      setError("אנא בחר קובץ מסוג CSV בלבד");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("לא נבחר קובץ");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const error = await res.json();
        setError(error.result || "שגיאה");
        return;
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);

      // Create a temporary link and click it
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "anonymized.csv");
      document.body.appendChild(link);
      link.click();
      link.remove();

      setResult("הקובץ הופק בהצלחה!");
      setError("");
    } catch (err) {
      console.error(err);
      setError("שגיאה בשליחת הקובץ לשרת");
      setResult("");
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 10, textAlign: "right" }}>
      <Typography variant="h5" gutterBottom>
        לביצוע אנונימיזציה CSV העלה קובץ
      </Typography>

      <Box component="form" onSubmit={handleSubmit} noValidate>
        <Input
          type="file"
          inputProps={{ accept: ".csv", dir: "rtl" }}
          onChange={handleFileChange}
          fullWidth
          sx={{ mb: 2 }}
        />
        <Button type="submit" variant="contained" fullWidth>
          שלח
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mt: 3 }}>
          {error}
        </Alert>
      )}

      {result && (
        <Typography variant="h6" sx={{ mt: 4 }}>
          תוצאה: {result}
        </Typography>
      )}
    </Container>
  );
}

export default App;
