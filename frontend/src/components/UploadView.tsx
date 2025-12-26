import React, { useState, useEffect } from 'react';

const UploadView: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [uploading, setUploading] = useState(false);
    const [processing, setProcessing] = useState(false);
    const [status, setStatus] = useState<string | null>(null);
    const [inspectionId, setInspectionId] = useState<number | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0];
            setFile(selectedFile);
            setPreviewUrl(URL.createObjectURL(selectedFile));
            setStatus(null);
            setInspectionId(null);
            setProcessing(false);
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setUploading(true);
        setStatus("Uploading Video...");

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/upload', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                setStatus("Upload complete. Processing initiated...");
                setInspectionId(data.inspection_id);
                setProcessing(true);
            } else {
                setStatus("Upload failed. Check server logs.");
                setUploading(false); // Only reset if failed
            }
        } catch (error) {
            console.error(error);
            setStatus("Network error during upload.");
            setUploading(false);
        }
        // Don't setUploading(false) here on success, keep it true to disable inputs during processing
    };

    // Polling Effect
    useEffect(() => {
        let interval: any;

        if (processing && inspectionId) {
            interval = setInterval(async () => {
                try {
                    const res = await fetch(`http://localhost:8000/inspections/${inspectionId}/status`);
                    if (res.ok) {
                        const data = await res.json();
                        if (data.status === 'COMPLETED') {
                            setStatus("Processing Complete! Check History Tab to view results. ✅");
                            setProcessing(false);
                            setUploading(false);
                            clearInterval(interval);
                        } else {
                            setStatus((prev) => prev === "Processing..." ? "Processing.. " : "Processing...");
                        }
                    }
                } catch (e) {
                    console.error("Polling error", e);
                }
            }, 2000); // Check every 2 seconds
        }

        return () => clearfix(interval);
    }, [processing, inspectionId]);

    // Helper to fix TS error with clearInterval
    const clearfix = (i: any) => clearInterval(i);

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Top Section: Upload & Preview */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-[400px]">

                {/* Upload Panel */}
                <div className={`bg-gray-900/50 backdrop-blur rounded-2xl p-6 flex flex-col items-center justify-center border-dashed border-2  transition-colors group relative ${uploading ? 'opacity-50 border-gray-800' : 'border-gray-700 hover:border-gray-500'}`}>
                    <input
                        type="file"
                        accept="video/*"
                        className="hidden"
                        id="video-upload"
                        onChange={handleFileChange}
                        disabled={uploading}
                    />
                    <label htmlFor="video-upload" className={`cursor-pointer text-center space-y-4 w-full h-full flex flex-col items-center justify-center relative z-10 ${uploading ? 'cursor-not-allowed' : ''}`}>
                        <div className="w-16 h-16 rounded-full bg-gray-800 group-hover:bg-gray-700 flex items-center justify-center transition-colors">
                            <span className="text-2xl">📤</span>
                        </div>
                        <div className="space-y-1">
                            <h3 className="text-xl font-bold text-white">
                                {file ? "Change Video" : "Upload Inspection Video"}
                            </h3>
                            <p className="text-gray-400 text-sm">
                                {file ? file.name : "Drag drop or click to browse"}
                            </p>
                        </div>
                    </label>
                </div>

                {/* Preview Panel (Loading State) */}
                <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-2xl p-6 relative overflow-hidden flex flex-col items-center justify-center">
                    {previewUrl ? (
                        <div className="w-full h-full flex flex-col gap-4">
                            <h3 className="text-sm font-bold text-gray-400 uppercase">Preview</h3>

                            {/* Video Wrapper with Overlay */}
                            <div className="flex-1 bg-black rounded-lg overflow-hidden border border-gray-700 relative">
                                <video src={previewUrl} controls className="w-full h-full object-contain" />

                                {processing && (
                                    <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center z-20 backdrop-blur-sm animate-in fade-in">
                                        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                                        <span className="text-blue-400 font-mono animate-pulse">AI PROCESSING PIPELINE RUNNING</span>
                                        <span className="text-xs text-gray-500 mt-2">Checking Wagons, OCR, Defects...</span>
                                    </div>
                                )}
                            </div>

                            <div className="flex justify-between items-center bg-gray-800/50 p-3 rounded-lg border border-white/5">
                                {status && (
                                    <span className={`text-sm font-bold ${status.includes("Complete") ? "text-green-400" : "text-yellow-400"}`}>
                                        {status}
                                    </span>
                                )}
                                {!processing && (
                                    <button
                                        onClick={handleUpload}
                                        disabled={uploading}
                                        className={`bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-lg font-bold transition-all ml-auto ${uploading ? 'hidden' : ''}`}
                                    >
                                        Start Processing 🚀
                                    </button>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div className="text-gray-600 italic">
                            Video preview will appear here
                        </div>
                    )}
                </div>
            </div>

            {/* Guidelines / Info */}
            <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-2xl p-6">
                <h3 className="text-lg font-bold text-white mb-4">How it works</h3>
                <ul className="list-disc list-inside text-gray-400 space-y-2">
                    <li>Upload an MP4/AVI file of the freight train inspection.</li>
                    <li>The system will automatically queue it for processing.</li>
                    <li>The pipeline includes: Night Detection, Zero-DCE Enhancement, Wagon Detection, OCR, and Deblurring.</li>
                    <li>You can monitor progress implicitly; once finished, results will appear in the <strong>History</strong> tab.</li>
                </ul>
            </div>
        </div>
    );
};

export default UploadView;

