import React, { useState } from "react";
import Lottie from "lottie-react";
import sparkles from "../assets/loadingAnimation.json"

const UploadAudio: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploadMessage, setUploadMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [summary, setSummary] = useState("");
  const [emotion, setEmotion] = useState("");
  const [importantPoints, setImportantPoints] = useState<string[]>([]);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];

      if (selectedFile.size > 30 * 1024 * 1024) {
        setUploadMessage("❌ File too large. Max allowed is 30MB.");
        return;
      }

      setFile(selectedFile);
      setUploadMessage("");
      setSummary("");
      setImportantPoints([]);

      const formData = new FormData();
      formData.append("file", selectedFile);

      setIsLoading(true);

      try {
        const res = await fetch(`${import.meta.env.VITE_API_BASE_URL}/upload`, {
          method: "POST",
          body: formData,
        });

        if (res.ok) {
          const data = await res.json();
          setUploadMessage("✅ File uploaded successfully!");
          setSummary(data.summary);
          setEmotion(data.emotion);
          setImportantPoints(data.important_points || []);
        } else {
          setUploadMessage("❌ Upload failed.");
          setIsLoading(false);
        }
      } catch (err) {
        setUploadMessage("❌ Error during upload.");
        setIsLoading(false);
      } finally {
        setIsLoading(false); // stop loader
      }
    }
  };

  return (
    <div className="w-full max-w-2xl flex flex-col items-center mb-10">
      <div className="relative p-6 rounded-xl shadow-xl w-full text-center pb-15">
        {isLoading && (
          <div className="absolute inset-0 bg-white bg-opacity-60 z-10 flex flex-col items-center justify-center rounded-xl">
            <Lottie animationData={sparkles} loop style={{ width: 100, height: 100 }} />
            <p className="mt-3 text-sm text-gray-700 font-medium animate-pulse">
              Processing your audio magically...
            </p>
          </div>
        )}
        <h2 className="text-xl font-bold mb-4">Upload Meeting Audio</h2>

        <div className="flex justify-center">
          <label
            htmlFor="audio-upload"
            className="cursor-pointer bg-blue-500 text-white font-medium py-5 px-10 rounded hover:bg-blue-600 transition duration-300"
          >
            Upload Audio
            <input
              id="audio-upload"
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>
        </div>

        {file && (
          <p className="mt-3 text-sm text-gray-700">
            Selected File: <span className="font-medium">{file.name}</span>
          </p>
        )}

        {uploadMessage && (
          <p className="mt-2 text-sm text-gray-600">{uploadMessage}</p>
        )}
      </div>

      {summary && (
        <div className="mt-6 bg-white p-6 rounded-xl shadow-lg w-full">
          <h3 className="text-lg font-bold text-gray-800 mb-2">📋 Summary</h3>
          <p className="font-Kanit text-10 text-gray-800  pl-7 leading-relaxed">{summary}</p>
        </div>
      )}

      {emotion && (
        <div className="mt-4 bg-white p-4 rounded-xl shadow-lg w-full">
          <h3 className="text-md font-bold text-gray-800 mb-1">😊 Emotion Insight</h3>
          <p className="font-Kanit text-10 text-gray-800  pl-7 leading-relaxed">{emotion}</p>
        </div>
      )}

      {importantPoints.length > 0 && (
        <div className="mt-4 bg-white p-4 rounded-xl shadow-lg w-full">
          <h3 className="text-md font-bold text-gray-900 mb-2">🔑 Important Points</h3>
          <ul className="list-disc pl-8 text-10 text-gray-800">
            {importantPoints.map((point, index) => (
              <li key={index}>{point}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default UploadAudio;
