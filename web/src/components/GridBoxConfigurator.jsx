import { useState } from 'react'

export default function GridBoxConfigurator({ onGridChange }) {
    const [method, setMethod] = useState('auto')
    const [center, setCenter] = useState({ x: 0, y: 0, z: 0 })
    const [size, setSize] = useState({ x: 20, y: 20, z: 20 })
    const [isDetecting, setIsDetecting] = useState(false)

    const detectBindingSite = async () => {
        setIsDetecting(true)
        // TODO: Call backend API to detect binding site
        setTimeout(() => {
            // Mock auto-detection (will be replaced with real API call)
            setCenter({ x: 15.0, y: 20.5, z: 10.2 })
            setSize({ x: 22, y: 22, z: 22 })
            setIsDetecting(false)
            updateGrid({ x: 15.0, y: 20.5, z: 10.2 }, { x: 22, y: 22, z: 22 })
        }, 1000)
    }

    const updateGrid = (newCenter, newSize) => {
        if (onGridChange) {
            onGridChange({
                grid_center_x: newCenter.x,
                grid_center_y: newCenter.y,
                grid_center_z: newCenter.z,
                grid_size_x: newSize.x,
                grid_size_y: newSize.y,
                grid_size_z: newSize.z
            })
        }
    }

    const handleCenterChange = (axis, value) => {
        const newCenter = { ...center, [axis]: parseFloat(value) }
        setCenter(newCenter)
        updateGrid(newCenter, size)
    }

    const handleSizeChange = (axis, value) => {
        const newSize = { ...size, [axis]: parseInt(value) }
        setSize(newSize)
        updateGrid(center, newSize)
    }

    const gridVolume = size.x * size.y * size.z
    const isOptimalSize = gridVolume >= 8000 && gridVolume <= 27000

    return (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
            <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z"></path>
                </svg>
                Grid Box Configuration
            </h3>

            {/* Method Selector */}
            <div className="mb-6">
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                    Detection Method
                </label>
                <div className="flex gap-2">
                    <select
                        value={method}
                        onChange={(e) => setMethod(e.target.value)}
                        className="flex-1 px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
                    >
                        <option value="auto">Auto-Detect (Center of Mass)</option>
                        <option value="ligand">From Existing Ligand</option>
                        <option value="manual">Manual Input</option>
                    </select>
                    <button
                        onClick={detectBindingSite}
                        disabled={isDetecting}
                        className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                        {isDetecting ? (
                            <>
                                <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Detecting...
                            </>
                        ) : (
                            <>
                                🎯 Detect
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* Grid Parameters */}
            <div className="grid md:grid-cols-2 gap-6">
                {/* Center Coordinates */}
                <div className="space-y-4">
                    <h4 className="font-semibold text-slate-700 text-sm uppercase tracking-wider">Center (Ų)</h4>

                    <div>
                        <label className="block text-xs font-medium text-slate-600 mb-1">X-axis</label>
                        <div className="flex items-center gap-3">
                            <input
                                type="range"
                                min="-50"
                                max="50"
                                step="0.5"
                                value={center.x}
                                onChange={(e) => handleCenterChange('x', e.target.value)}
                                className="flex-1"
                            />
                            <input
                                type="number"
                                value={center.x}
                                onChange={(e) => handleCenterChange('x', e.target.value)}
                                className="w-20 px-2 py-1 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
                                step="0.5"
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-xs font-medium text-slate-600 mb-1">Y-axis</label>
                        <div className="flex items-center gap-3">
                            <input
                                type="range"
                                min="-50"
                                max="50"
                                step="0.5"
                                value={center.y}
                                onChange={(e) => handleCenterChange('y', e.target.value)}
                                className="flex-1"
                            />
                            <input
                                type="number"
                                value={center.y}
                                onChange={(e) => handleCenterChange('y', e.target.value)}
                                className="w-20 px-2 py-1 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
                                step="0.5"
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-xs font-medium text-slate-600 mb-1">Z-axis</label>
                        <div className="flex items-center gap-3">
                            <input
                                type="range"
                                min="-50"
                                max="50"
                                step="0.5"
                                value={center.z}
                                onChange={(e) => handleCenterChange('z', e.target.value)}
                                className="flex-1"
                            />
                            <input
                                type="number"
                                value={center.z}
                                onChange={(e) => handleCenterChange('z', e.target.value)}
                                className="w-20 px-2 py-1 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
                                step="0.5"
                            />
                        </div>
                    </div>
                </div>

                {/* Box Size */}
                <div className="space-y-4">
                    <h4 className="font-semibold text-slate-700 text-sm uppercase tracking-wider">Size (Ų)</h4>

                    <div>
                        <label className="block text-xs font-medium text-slate-600 mb-1">X dimension</label>
                        <div className="flex items-center gap-3">
                            <input
                                type="range"
                                min="10"
                                max="40"
                                step="1"
                                value={size.x}
                                onChange={(e) => handleSizeChange('x', e.target.value)}
                                className="flex-1"
                            />
                            <input
                                type="number"
                                value={size.x}
                                onChange={(e) => handleSizeChange('x', e.target.value)}
                                className="w-20 px-2 py-1 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-xs font-medium text-slate-600 mb-1">Y dimension</label>
                        <div className="flex items-center gap-3">
                            <input
                                type="range"
                                min="10"
                                max="40"
                                step="1"
                                value={size.y}
                                onChange={(e) => handleSizeChange('y', e.target.value)}
                                className="flex-1"
                            />
                            <input
                                type="number"
                                value={size.y}
                                onChange={(e) => handleSizeChange('y', e.target.value)}
                                className="w-20 px-2 py-1 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-xs font-medium text-slate-600 mb-1">Z dimension</label>
                        <div className="flex items-center gap-3">
                            <input
                                type="range"
                                min="10"
                                max="40"
                                step="1"
                                value={size.z}
                                onChange={(e) => handleSizeChange('z', e.target.value)}
                                className="flex-1"
                            />
                            <input
                                type="number"
                                value={size.z}
                                onChange={(e) => handleSizeChange('z', e.target.value)}
                                className="w-20 px-2 py-1 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* Grid Info */}
            <div className="mt-6 p-4 bg-slate-50 rounded-lg border border-slate-200">
                <div className="flex items-start gap-3">
                    <svg className={`w-5 h-5 flex-shrink-0 mt-0.5 ${isOptimalSize ? 'text-green-600' : 'text-amber-600'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <div className="flex-1">
                        <div className="text-sm font-medium text-slate-900 mb-1">
                            Grid Volume: {gridVolume.toLocaleString()} Ųł
                        </div>
                        <div className={`text-xs ${isOptimalSize ? 'text-green-700' : 'text-amber-700'}`}>
                            {isOptimalSize ? (
                                '✓ Optimal size for efficient docking (8,000 - 27,000 Ųł)'
                            ) : gridVolume < 8000 ? (
                                '⚠️ Box may be too small. Consider increasing size for better coverage.'
                            ) : (
                                '⚠️ Box may be too large. This will increase computation time significantly.'
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Tips */}
            <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <div className="flex gap-2">
                    <svg className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <div className="text-xs text-blue-900">
                        <strong>Tip:</strong> The grid box should encompass the binding site with some padding.
                        Use auto-detect for a quick start, then fine-tune manually if needed.
                    </div>
                </div>
            </div>
        </div>
    )
}
