import React from 'react';

interface VideoFeedProps {
    streamId: number;
}

const VideoFeed: React.FC<VideoFeedProps> = ({ streamId }) => {
    const [isProcessing, setIsProcessing] = React.useState(false);

    const toggleProcessing = async () => {
        try {
            if (isProcessing) {
                await fetch('http://localhost:8000/live/stop', { method: 'POST' });
                setIsProcessing(false);
            } else {
                await fetch(`http://localhost:8000/live/start?stream_id=${streamId}`, { method: 'POST' });
                setIsProcessing(true);
            }
        } catch (error) {
            console.error("Failed to toggle processing:", error);
        }
    };

    // Check status on mount
    React.useEffect(() => {
        fetch('http://localhost:8000/live/status')
            .then(res => res.json())
            .then(data => {
                if (data.is_running) setIsProcessing(true);
            })
            .catch(console.error);
    }, []);

    return (
        <div className="relative w-full aspect-video rounded-2xl overflow-hidden shadow-2xl border border-gray-700 bg-black">
            {/* Live Indicator */}
            <div className="absolute top-4 left-4 z-10 flex items-center gap-2 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full border border-red-500/30">
                <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
                <span className="text-xs font-bold text-red-500 tracking-wider">LIVE</span>
            </div>

            {/* Overlay Information & Controls */}
            <div className="absolute top-4 right-4 z-10 flex flex-col items-end gap-2">
                <div className="bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10">
                    <span className="text-xs font-medium text-gray-400">CAM-0{streamId} • 1080p</span>
                </div>

                {/* Processing Toggle */}
                <button
                    onClick={toggleProcessing}
                    className={`
                        flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-bold tracking-wide transition-all
                        ${isProcessing
                            ? 'bg-red-500/20 text-red-500 border border-red-500/50 hover:bg-red-500/30'
                            : 'bg-green-500/20 text-green-500 border border-green-500/50 hover:bg-green-500/30'}
                    `}
                >
                    <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`} />
                    {isProcessing ? 'STOP AI' : 'START AI'}
                </button>
            </div>

            {/* Video Stream */}
            <img
                src={`http://localhost:8000/video_feed/${streamId}?t=${Date.now()}`}
                alt={`Stream ${streamId}`}
                className="w-full h-full object-cover"
            />

            {/* Processing Overlay (Active State) */}
            {isProcessing && (
                <div className="absolute inset-0 pointer-events-none border-2 border-green-500/40 animate-pulse">
                    <div className="absolute bottom-10 left-1/2 -translate-x-1/2 bg-black/70 backdrop-blur-md px-6 py-2 rounded-full border border-green-500/50 flex items-center gap-3">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" />
                        <span className="text-xs font-bold text-green-400">AI PIPELINE ACTIVE • DETECTING WAGONS</span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default VideoFeed;