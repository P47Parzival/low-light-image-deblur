import React, { useState } from 'react';

const UploadView: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Top Section: Upload & Stats */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-[400px]">

                {/* Upload Panel */}
                <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-2xl p-6 flex flex-col items-center justify-center border-dashed border-2 border-gray-700 hover:border-gray-500 transition-colors group">
                    <input
                        type="file"
                        accept="video/*"
                        className="hidden"
                        id="video-upload"
                        onChange={handleFileChange}
                    />
                    <label htmlFor="video-upload" className="cursor-pointer text-center space-y-4 w-full h-full flex flex-col items-center justify-center">
                        <div className="w-16 h-16 rounded-full bg-gray-800 group-hover:bg-gray-700 flex items-center justify-center transition-colors">
                            <span className="text-2xl">📤</span>
                        </div>
                        <div className="space-y-1">
                            <h3 className="text-xl font-bold text-white">
                                {file ? file.name : "Upload Inspection Video"}
                            </h3>
                            <p className="text-gray-400 text-sm">
                                {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "Drag drop or click to browse"}
                            </p>
                        </div>
                    </label>
                </div>

                {/* Info Panel (Static Data) */}
                <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-2xl p-6 relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
                        <span className="text-9xl font-mono font-bold text-white">DATA</span>
                    </div>

                    <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                        Analysis Results
                    </h3>

                    <div className="space-y-4 font-mono text-sm">
                        <div className="bg-white/5 p-4 rounded-lg border border-white/5 space-y-3">
                            <div className="flex justify-between items-center border-b border-white/5 pb-2">
                                <span className="text-gray-400">Wagon ID:</span>
                                <span className="text-white font-bold bg-blue-500/20 px-2 py-0.5 rounded border border-blue-500/30">
                                    RAIL-84922
                                </span>
                            </div>

                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">Condition:</span>
                                <div className="flex gap-2">
                                    <span className="text-xs bg-purple-500/20 text-purple-400 px-2 py-0.5 rounded border border-purple-500/30">
                                        Night Detected
                                    </span>
                                    <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded border border-red-500/30">
                                        Damage Found
                                    </span>
                                </div>
                            </div>

                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">Action:</span>
                                <span className="text-yellow-400 font-bold animate-pulse">
                                    MANUAL CHECK REQ
                                </span>
                            </div>

                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">OCR Result:</span>
                                <span className="text-green-400 font-bold">
                                    21110867659
                                </span>
                            </div>

                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">Blur Mitigation:</span>
                                <span className="text-gray-500 font-bold">
                                    NOT REQUIRED
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Bottom Section: Media Inspection */}
            <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-2xl p-6">
                <h3 className="text-lg font-bold text-white mb-6">Visual Forensics</h3>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Day vs Night */}
                    <div className="space-y-3">
                        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Environment Analysis</h4>
                        <div className="grid grid-rows-2 gap-3 h-[300px]">
                            <div className="bg-black/40 rounded-lg border border-white/10 flex items-center justify-center relative group overflow-hidden">
                                <span className="text-xs text-gray-500 italic z-10">Day Reference</span>
                                <div className="absolute inset-0 bg-linear-to-b from-transparent to-black/80"></div>
                            </div>
                            <div className="bg-black/40 rounded-lg border border-white/10 flex items-center justify-center relative group overflow-hidden border-purple-500/30">
                                <span className="text-xs text-purple-400 italic z-10">Night (Source)</span>
                                <div className="absolute top-2 right-2 px-2 py-1 bg-purple-500/20 text-purple-300 text-xs rounded border border-purple-500/30">
                                    Low Light
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* OCR Frame */}
                    <div className="space-y-3">
                        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Number Extraction</h4>
                        <div className="h-[300px] bg-black/40 rounded-lg border border-white/10 flex flex-col items-center justify-center p-4 space-y-4">
                            <div className="w-full flex-1 bg-gray-800/50 rounded flex items-center justify-center border border-white/5">
                                <span className="text-xs text-gray-600">Cropped Number Plate</span>
                            </div>
                            <div className="w-full p-4 bg-gray-800 rounded-lg border border-gray-700 text-center">
                                <span className="text-xs text-gray-400 block mb-1">Detected Sequence</span>
                                <span className="text-2xl font-mono font-bold text-white tracking-[0.2em] selection:bg-blue-500/30">
                                    21110867659
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Blur vs Deblur */}
                    <div className="space-y-3">
                        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Blur Mitigation</h4>
                        <div className="grid grid-rows-2 gap-3 h-[300px]">
                            <div className="bg-black/40 rounded-lg border border-white/10 flex items-center justify-center relative">
                                <span className="text-xs text-gray-500 italic">Original Input</span>
                            </div>
                            <div className="bg-black/40 rounded-lg border border-white/10 flex items-center justify-center relative">
                                <span className="text-xs text-gray-600 italic">Enhanced Output</span>
                                <div className="absolute inset-0 flex items-center justify-center backdrop-blur-[2px]">
                                    <span className="bg-gray-800/80 px-3 py-1 rounded-full text-xs text-gray-400 border border-gray-700">
                                        Enhancement Skipped
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default UploadView;
