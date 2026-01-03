import React, { useState, useEffect } from 'react';
import { Upload } from 'lucide-react';

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
            <div className="flex items-center gap-4 mb-8">
                <div className="w-16 h-0.5 bg-gradient-to-r from-blue-500 to-transparent"></div>
                <span className="text-xs tracking-[0.3em] text-blue-400 font-bold uppercase">Upload Module</span>
            </div>

            {/* If no file is selected, show the big upload button */}
            {!file && (
                <div className="border-2 border-dashed border-zinc-800 rounded-2xl p-16 text-center hover:border-zinc-700 transition-all bg-zinc-950/50 relative">
                     <input
                        type="file"
                        accept="video/*"
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        id="video-upload"
                        onChange={handleFileChange}
                        disabled={uploading}
                    />
                    <label htmlFor="video-upload" className="cursor-pointer">
                        <Upload className="w-16 h-16 text-zinc-600 mx-auto mb-6" />
                        <h3 className="text-xl font-bold text-white mb-2">Upload Video Files</h3>
                        <p className="text-zinc-500 mb-6">Drag and drop your inspection videos here</p>
                        <span className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg">
                            Select Files
                        </span>
                    </label>
                </div>
            )}

            {/* If a file is selected, show preview and controls */}
            {previewUrl && (
                 <div className="bg-zinc-950/70 border border-zinc-800 rounded-2xl p-6 relative overflow-hidden flex flex-col gap-4">
                    <h3 className="text-sm font-bold text-zinc-400 uppercase">Preview &amp; Process</h3>

                    {/* Video Wrapper with Overlay */}
                    <div className="flex-1 bg-black rounded-lg overflow-hidden border border-zinc-700 relative aspect-video">
                        <video src={previewUrl} controls className="w-full h-full object-contain" />

                        {processing && (
                            <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center z-20 backdrop-blur-sm animate-in fade-in">
                                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                                <span className="text-blue-400 font-mono animate-pulse">AI PROCESSING PIPELINE RUNNING</span>
                                <span className="text-xs text-zinc-500 mt-2">Checking Wagons, OCR, Defects...</span>
                            </div>
                        )}
                    </div>

                    <div className="flex justify-between items-center bg-zinc-900/50 p-3 rounded-lg border border-zinc-800">
                        <div className='flex flex-col'>
                            <span className={`text-sm font-bold ${status && status.includes("Complete") ? "text-emerald-400" : "text-amber-400"}`}>
                                {status || file?.name}
                            </span>
                            {status && <span className='text-xs text-zinc-500'>{file?.name}</span>}
                        </div>
                        {!processing && (
                            <button
                                onClick={handleUpload}
                                disabled={uploading}
                                className={`bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white px-6 py-2 rounded-lg font-semibold transition-all ml-auto ${uploading ? 'hidden' : ''}`}
                            >
                                Start Processing
                            </button>
                        )}
                    </div>
                </div>
            )}

            {/* Guidelines / Info */}
            <div className="bg-zinc-950/50 border border-zinc-800 rounded-2xl p-6">
                <h3 className="text-lg font-bold text-white mb-4">How it works</h3>
                <ul className="list-disc list-inside text-zinc-400 space-y-2">
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
