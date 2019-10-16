import sys


def swc2obj(in_path, output_path):
    """
    Convert swc file format into obj with only Geometric vertices (v) and Line element (l)

    See http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    for swc format specification.

    Parameters
    ----------
    in_path: str
        Path for input swc file
    output_path: str
        Path for output the obj file
    """

    # Parse the SWC format
    x = open(in_path, 'r')
    swc = {}
    for line in x:
        if not line.startswith('#') and not line == '\n':
            splits = line.split()
            # Extract the index
            index = int(splits[0])
            # Extract the type
            n_type = int(splits[1])
            # Extract the coordinate
            x = float(splits[2])
            y = float(splits[3])
            z = float(splits[4])
            # Extract the radius
            r = float(splits[5])
            # Extract the parent index
            parent = int(splits[-1])
            swc[index] = (float(x), float(y), float(z), r, parent, n_type)
    # Sort the index since obj need to have sorted index in order to write the lines
    sorted_index = sorted(swc.keys())
    # Write the obj file
    fid = open(output_path, 'w')
    # Store the coordinate of each node
    for index in sorted_index:
        node = swc[index]
        fid.write('v {} {} {}\n'.format(node[0], node[1], node[2]))
    # Store the relationship between nodes (current to parent)
    for index in sorted_index:
        if index < 2:
            continue
        node = swc[index]
        parent_point_index = node[4]
        if parent_point_index == -1:  # Not parent, first node
            continue
        fid.write('l {} {}\n'.format(index, parent_point_index))
    fid.write('\n')
    fid.close()
