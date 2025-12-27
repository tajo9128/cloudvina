
import sys
import os

# Mock dependencies if strictly needed, but Layer1Generator is mostly pure logic + requests
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'api'))

from services.layer1_generator import Layer1Generator

def test_layer1():
    print("=== Layer 1 Generator Isolation Test ===")
    
    # 1. Create Dummy Crystal PDB
    test_pdb = "test_crystal.pdb"
    with open(test_pdb, "w") as f:
        # Minimal valid PDB line for NMA parser
        f.write("ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N\n")
        f.write("ATOM      2  CA  ALA A   1      11.500  11.500  11.500  1.00  0.00           C\n")
    
    # 2. Initialize Generator
    # Note: AlphaFold fetch will fail (return None) because we don't have a real PDB ID match logic for "test_crystal"
    # But NMA should work (mock perturbation)
    generator = Layer1Generator(test_pdb, "job_test_123", use_af=True, use_nma=True)
    
    # 3. Generate
    ensemble = generator.generate()
    
    print(f"\nEnsemble Size: {len(ensemble)}")
    for i, path in enumerate(ensemble):
        print(f"Structure {i}: {path}")
    
    # Validation
    assert len(ensemble) >= 2, "Should have at least Crystal + NMA"
    assert ensemble[0] == test_pdb, "First must be Crystal"
    assert "_nma" in ensemble[-1] or "_nma" in ensemble[1], "Should contain NMA variant"
    
    print("\n[OK] Verification SUCCESS: Layer 1 Generator logic works.")
    
    # Cleanup
    try:
        os.remove(test_pdb)
        for p in ensemble:
            if p != test_pdb and os.path.exists(p):
                os.remove(p)
    except:
        pass

if __name__ == "__main__":
    test_layer1()
