import React from "react";
import UploadAudio from "../components/UploadAudio";

const HomePage: React.FC = () => {
  return (
    <>
      <div className="min-h-screen flex flex-col items-center p-6">
        <header className="mb-8 text-center">
          <h1 className="font-ArchivoBlack text-5xl text-blue-900">MeetingMind</h1>
          <p className="text-blue-900 text-1xl m-3">
            AI-powered meeting summaries and emotion insights
          </p>
        </header>

        <div className="w-full max-w-7xl flex justify-center overflow-hidden">
          <UploadAudio />
        </div>
      </div>

    </>
  );
};

export default HomePage;
