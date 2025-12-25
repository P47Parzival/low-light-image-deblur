import React from 'react';

interface NavbarProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
}

const Navbar: React.FC<NavbarProps> = ({ activeTab, onTabChange }) => {
    const tabs = ["Live", "Upload", "History"];

    return (
        <nav className="fixed top-0 w-full z-50 bg-black/50 backdrop-blur-xl border-b border-white/5">
            <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                {/* Brand */}
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-linear-to-tr from-blue-600 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                        <span className="font-bold text-white text-lg">M</span>
                    </div>
                    <span className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-white to-gray-400">
                        Garud
                    </span>
                </div>

                {/* Navigation Tabs */}
                <div className="flex items-center gap-1 bg-white/5 p-1 rounded-lg border border-white/5">
                    {tabs.map((tab) => (
                        <button
                            key={tab}
                            onClick={() => onTabChange(tab)}
                            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all duration-200 ${activeTab === tab
                                    ? 'bg-blue-600 text-white shadow-md shadow-blue-500/20'
                                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                                }`}
                        >
                            {tab}
                        </button>
                    ))}
                </div>

                {/* Status / User */}
                <div className="flex items-center gap-6">
                    <div className="text-xs font-mono text-gray-400 hidden md:block">
                        SERVER: <span className="text-green-400">ONLINE</span>
                    </div>
                    <div className="w-8 h-8 rounded-full bg-gray-800 border border-gray-700"></div>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
