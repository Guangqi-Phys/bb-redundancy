from itertools import combinations

def set_xor(set1, set2):
    """
    Compute XOR (symmetric difference) of two sets.
    Returns elements that are in either set1 or set2, but not in both.
    """
    return set1.symmetric_difference(set2)

def find_all_xor_dependencies(dependencies):
    """
    Find all possible XOR combinations of dependency sets to generate shorter dependencies.
    Each dependency set includes the redundant row itself.
    
    Args:
        dependencies (dict): Dictionary mapping redundant row indices to lists of dependencies
    
    Returns:
        list: All XOR combinations with their analysis
    """
    redundant_rows = list(dependencies.keys())
    # Include each redundant row in its own dependency set
    dependency_sets = {row: set(deps + [row]) for row, deps in dependencies.items()}
    
    print("Original dependency sets (including redundant row itself):")
    print("=" * 60)
    for row, deps in dependencies.items():
        full_set = dependency_sets[row]
        print(f"Row {row}: {sorted(list(full_set))} (size: {len(full_set)})")
    print()
    
    all_xor_results = []
    
    # Compute XOR for all pairs
    print("XOR of pairs:")
    print("=" * 40)
    for i, (row1, row2) in enumerate(combinations(redundant_rows, 2), 1):
        set1 = dependency_sets[row1]
        set2 = dependency_sets[row2]
        xor_result = set_xor(set1, set2)
        
        result_info = {
            'type': 'pair_xor',
            'rows': [row1, row2],
            'original_sizes': [len(set1), len(set2)],
            'xor_set': sorted(list(xor_result)),
            'xor_size': len(xor_result),
            'common_elements': sorted(list(set1.intersection(set2))),
            'common_count': len(set1.intersection(set2))
        }
        
        all_xor_results.append(result_info)
        
        print(f"{i:2d}. Row {row1} XOR Row {row2}:")
        print(f"    Set1: {sorted(list(set1))} (size {len(set1)})")
        print(f"    Set2: {sorted(list(set2))} (size {len(set2)})")
        print(f"    Common: {sorted(list(set1.intersection(set2)))} (size {len(set1.intersection(set2))})")
        print(f"    XOR: {sorted(list(xor_result))} (size {len(xor_result)})")
        print()
    
    # Compute XOR for all triples
    print("\nXOR of triples:")
    print("=" * 40)
    triple_count = 0
    for row1, row2, row3 in combinations(redundant_rows, 3):
        triple_count += 1
        set1 = dependency_sets[row1]
        set2 = dependency_sets[row2]
        set3 = dependency_sets[row3]
        
        # XOR of three sets: ((set1 XOR set2) XOR set3)
        xor_12 = set_xor(set1, set2)
        xor_result = set_xor(xor_12, set3)
        
        result_info = {
            'type': 'triple_xor',
            'rows': [row1, row2, row3],
            'original_sizes': [len(set1), len(set2), len(set3)],
            'xor_set': sorted(list(xor_result)),
            'xor_size': len(xor_result)
        }
        
        all_xor_results.append(result_info)
        
        print(f"{triple_count:2d}. Row {row1} XOR Row {row2} XOR Row {row3}:")
        print(f"    XOR result: {sorted(list(xor_result))} (size {len(xor_result)})")
        print()
    
    # Compute XOR for all 4-tuples, 5-tuples, and 6-tuple
    print("\nXOR of larger combinations:")
    print("=" * 40)
    
    for r in range(4, len(redundant_rows) + 1):
        combo_count = 0
        for combo in combinations(redundant_rows, r):
            combo_count += 1
            
            # Compute XOR iteratively
            result_set = dependency_sets[combo[0]]
            for row in combo[1:]:
                result_set = set_xor(result_set, dependency_sets[row])
            
            result_info = {
                'type': f'{r}_tuple_xor',
                'rows': list(combo),
                'original_sizes': [len(dependency_sets[row]) for row in combo],
                'xor_set': sorted(list(result_set)),
                'xor_size': len(result_set)
            }
            
            all_xor_results.append(result_info)
            
            rows_str = ' XOR '.join(f"Row {row}" for row in combo)
            print(f"{combo_count:2d}. {rows_str}:")
            print(f"    XOR result: {sorted(list(result_set))} (size {len(result_set)})")
            print()
    
    # Find the shortest XOR results
    print("\nSHORTEST XOR RESULTS:")
    print("=" * 40)
    
    # Sort by XOR size
    sorted_results = sorted(all_xor_results, key=lambda x: x['xor_size'])
    
    # Group by size
    by_size = {}
    for result in sorted_results:
        size = result['xor_size']
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(result)
    
    print(f"XOR results grouped by size:")
    for size in sorted(by_size.keys()):
        count = len(by_size[size])
        print(f"  Size {size}: {count} result(s)")
    
    print()
    
    # Show the shortest non-empty results (up to 6 shortest)
    print(f"THE 6 SHORTEST DEPENDENCY SETS FROM XOR OPERATIONS:")
    print("=" * 55)
    
    shortest_results = []
    for size in sorted(by_size.keys()):
        if size > 0:  # Only non-empty sets
            shortest_results.extend(by_size[size])
            if len(shortest_results) >= 6:
                break
    
    for i, result in enumerate(shortest_results[:6], 1):
        rows_str = ' XOR '.join(f"Row {r}" for r in result['rows'])
        print(f"{i}. {rows_str}: {result['xor_set']} (size {result['xor_size']})")
    
    return all_xor_results

def analyze_xor_dependencies():
    """
    Analyze XOR combinations of the specific dependencies from terminal output.
    """
    # Your 6 dependency sets (WITHOUT including the redundant row yet)
    specific_deps = {
        22: [0, 1, 3, 4, 6, 7, 9, 10, 18, 19, 21, 24, 25, 27, 28],
        23: [0, 2, 3, 5, 6, 8, 9, 11, 18, 20, 21, 24, 26, 27, 29],
        32: [2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 19, 21, 25, 29, 30],
        33: [1, 5, 6, 8, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 31],
        34: [1, 5, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 21, 24, 26, 30],
        35: [1, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 15, 18, 20, 24, 28, 31]
    }
    
    print("FINDING 6 SHORTER DEPENDENCY SETS THROUGH XOR OPERATIONS")
    print("=" * 70)
    print("\nInput: 6 dependency sets (each including its redundant row)")
    print("Output: 6 shortest XOR combinations representing new dependencies")
    print()
    
    xor_results = find_all_xor_dependencies(specific_deps)
    
    # Summary statistics
    original_sizes = [len(deps) + 1 for deps in specific_deps.values()]  # +1 for redundant row
    xor_sizes = [r['xor_size'] for r in xor_results if r['xor_size'] > 0]
    
    print(f"\nSUMMARY:")
    print("=" * 20)
    print(f"Original dependency sizes (with redundant row): {sorted(original_sizes)}")
    print(f"Average original size: {sum(original_sizes)/len(original_sizes):.1f}")
    if xor_sizes:
        print(f"XOR result sizes: {sorted(set(xor_sizes))}")
        print(f"Shortest XOR result: {min(xor_sizes)}")
        print(f"Number of XOR results: {len(xor_sizes)}")
        
        # Count how many are shorter than original
        min_original = min(original_sizes)
        shorter_count = sum(1 for size in xor_sizes if size < min_original)
        print(f"XOR results shorter than shortest original ({min_original}): {shorter_count}")
    
    return xor_results

if __name__ == "__main__":
    analyze_xor_dependencies()