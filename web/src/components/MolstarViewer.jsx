import { useEffect, useRef, useState } from 'react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui';
import { DefaultPluginUISpec } from 'molstar/lib/mol-plugin-ui/spec';
import { PluginConfig } from 'molstar/lib/mol-plugin/config';
import { StructureRepresentationPresetProvider } from 'molstar/lib/mol-plugin-state/builder/structure/representation-preset';
// Note: Molstar styles removed due to SCSS compatibility issues
// Using custom inline styles instead

export default function MolstarViewer({
    pdbqtData,
    receptorData,
    interactions = null,
    title = "3D Structure",
    width = "100%",
    height = "100%",
    bindingAffinity = null
}) {
    const containerRef = useRef(null);
    const pluginRef = useRef(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    // Initialize Molstar Plugin
    useEffect(() => {
        if (!containerRef.current || pluginRef.current) return;

        const initPlugin = async () => {
            try {
                const spec = DefaultPluginUISpec();

                // Configure plugin UI
                spec.config = [
                    [PluginConfig.Viewport.ShowExpand, false],
                    [PluginConfig.Viewport.ShowControls, true],
                    [PluginConfig.Viewport.ShowSettings, false],
                    [PluginConfig.Viewport.ShowSelectionMode, true],
                    [PluginConfig.Viewport.ShowAnimation, false]
                ];

                const plugin = await createPluginUI({
                    target: containerRef.current,
                    spec,
                    render: {
                        style: {
                            backgroundColor: '#ffffff'
                        }
                    }
                });

                pluginRef.current = plugin;
                setIsLoading(false);
            } catch (err) {
                console.error('Failed to initialize Molstar:', err);
                setError('Failed to initialize 3D viewer');
                setIsLoading(false);
            }
        };

        initPlugin();

        return () => {
            if (pluginRef.current) {
                pluginRef.current.dispose();
                pluginRef.current = null;
            }
        };
    }, []);

    // Load structures when data changes
    useEffect(() => {
        if (!pluginRef.current || isLoading) return;

        const loadStructures = async () => {
            try {
                const plugin = pluginRef.current;

                // Clear previous structures
                await plugin.clear();

                // Load receptor (protein)
                if (receptorData) {
                    const receptorDataObj = await plugin.builders.data.rawData({
                        data: receptorData,
                        label: 'Receptor'
                    });

                    const receptorTrajectory = await plugin.builders.structure.parseTrajectory(
                        receptorDataObj,
                        'pdb' // Molstar auto-detects format, but we hint PDB
                    );

                    const receptorModel = await plugin.builders.structure.createModel(receptorTrajectory);
                    const receptorStructure = await plugin.builders.structure.createStructure(receptorModel);

                    // Apply cartoon representation for protein
                    await plugin.builders.structure.representation.addRepresentation(receptorStructure, {
                        type: 'cartoon',
                        colorTheme: { name: 'secondary-structure' },
                        sizeTheme: { name: 'uniform' }
                    });
                }

                // Load ligand
                if (pdbqtData) {
                    const ligandDataObj = await plugin.builders.data.rawData({
                        data: pdbqtData,
                        label: 'Ligand'
                    });

                    const ligandTrajectory = await plugin.builders.structure.parseTrajectory(
                        ligandDataObj,
                        'pdbqt'
                    );

                    const ligandModel = await plugin.builders.structure.createModel(ligandTrajectory);
                    const ligandStructure = await plugin.builders.structure.createStructure(ligandModel);

                    // Apply ball-and-stick representation for ligand
                    await plugin.builders.structure.representation.addRepresentation(ligandStructure, {
                        type: 'ball-and-stick',
                        colorTheme: { name: 'element-symbol' },
                        sizeTheme: { name: 'uniform', value: 0.3 }
                    });

                    // Focus camera on ligand
                    const ligandData = ligandStructure.cell.obj?.data;
                    if (ligandData) {
                        await plugin.canvas3d.camera.focus(
                            ligandData,
                            { durationMs: 500 }
                        );
                    }
                } else if (receptorData) {
                    // If no ligand, focus on receptor
                    await plugin.canvas3d.camera.reset();
                }

            } catch (err) {
                console.error('Failed to load structures:', err);
                setError('Failed to load molecular structure');
            }
        };

        loadStructures();
    }, [pdbqtData, receptorData, isLoading]);

    return (
        <div className="relative w-full h-full bg-white rounded-xl overflow-hidden border border-slate-200 shadow-sm">
            <div ref={containerRef} className="absolute inset-0" />

            {/* Loading Overlay */}
            {isLoading && (
                <div className="absolute inset-0 bg-white/90 backdrop-blur flex items-center justify-center z-50">
                    <div className="text-center">
                        <div className="inline-block w-8 h-8 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mb-2"></div>
                        <p className="text-sm text-slate-600 font-medium">Loading 3D Viewer...</p>
                    </div>
                </div>
            )}

            {/* Error Overlay */}
            {error && (
                <div className="absolute inset-0 bg-white flex items-center justify-center z-50">
                    <div className="text-center px-4">
                        <div className="text-red-600 mb-2">⚠</div>
                        <p className="text-sm text-slate-700">{error}</p>
                    </div>
                </div>
            )}

            {/* Title Overlay */}
            {!isLoading && !error && title && (
                <div className="absolute top-4 left-4 z-10">
                    <h3 className="text-sm font-bold text-slate-700 bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg shadow-sm border border-slate-200">
                        {title}
                    </h3>
                </div>
            )}

            {/* Binding Affinity Badge */}
            {!isLoading && !error && bindingAffinity && (
                <div className="absolute top-4 right-4 z-10">
                    <div className="bg-indigo-600/90 backdrop-blur text-white px-3 py-1.5 rounded-lg shadow-sm text-xs font-bold">
                        Affinity: {bindingAffinity.toFixed(1)} kcal/mol
                    </div>
                </div>
            )}
        </div>
    );
}
