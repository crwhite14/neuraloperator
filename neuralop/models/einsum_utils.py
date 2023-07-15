import torch
import opt_einsum
import tensorly as tl
tl.set_backend('pytorch')


def einsum_complexhalf_two_input(eq, a, b):
    """
    Return the einsum(eq, a, b)
    We call this instead of standard einsum when either a or b is ComplexHalf,
    to run the operation with half precision.
    """
    assert len(eq.split(',')) == 2, "Einsum equation must have two inputs"

    # cast both tensors to real and half precision
    a = torch.view_as_real(a)
    b = torch.view_as_real(b)
    a = a.half()
    b = b.half()

    # create a new einsum equation 
    input_output = eq.split('->')
    new_output = 'xy' + input_output[1]
    input_terms = input_output[0].split(',')
    new_inputs = [input_terms[0] + 'x', input_terms[1] + 'y']
    new_eqn = new_inputs[0] + ',' + new_inputs[1] + '->' + new_output

    tmp = torch.einsum(new_eqn, a, b)
    res = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1)
    return torch.view_as_complex(res)

def einsum_complexhalf(eq, *args):
    """Compute einsum for complexhalf tensors"""
    optimized = True
    if optimized:
        return einsum_complexhalf_optimized(eq, *args)

    if len(args) == 2:
        return einsum_complexhalf_two_input(eq, *args)

    # todo: this can be made general. Call opt_einsum to get the partial_eqns
    assert eq == 'abcd,e,be,fe,ce,de->afcd', "Currently only implemented for this eqn"

    partial_eqns = ['fe,e->fe',
                    'de,be->deb',
                    'fe,ce->fec',
                    'fec,deb->fcdb',
                    'fcdb,abcd->afcd']

    tensors = {}
    labels = eq.split('->')[0].split(',')
    tensors = dict(zip(labels,args))

    for key, tensor in tensors.items():
        tensor = torch.view_as_real(tensor)
        tensor = tensor.half()
        tensors[key] = tensor

    # now all tensors are in the "view as real" form
    for partial_eq in partial_eqns:

        # get the tensors
        in_labels, out_label = partial_eq.split('->')
        in_labels = in_labels.split(',')
        in_tensors = [tensors[label] for label in in_labels]

        # create a new einsum equation 
        input_output = partial_eq.split('->')
        new_output = 'xy' + input_output[1]
        input_terms = input_output[0].split(',')
        new_inputs = [input_terms[0] + 'x', input_terms[1] + 'y']
        new_eqn = new_inputs[0] + ',' + new_inputs[1] + '->' + new_output

        # perform the einsum, and convert to "view as real" form
        tmp = torch.einsum(new_eqn, *in_tensors)
        result = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1)
        tensors[out_label] = result

    return torch.view_as_complex(tensors['afcd'])

def einsum_complexhalf_optimized(eq, *args):
    """
    Optimized version of einsum_complexhalf, specific to 2d Navier Stokes with cp factorization
    """
    if len(args) == 2:
        # if there are two inputs, it is faster to call this method
        return einsum_complexhalf_two_input(eq, *args)

    # hard-coded optimal path
    partial_eqns = ['ce,e->ce', 'de,be->deb', 'ce,fe->cef', 'cef,deb->cfdb', 'cfdb,abcd->afcd']

    # create a dict of the input tensors by their label in the einsum equation
    tensors = {}
    input_labels = eq.split('->')[0].split(',')
    output_label = eq.split('->')[1]
    tensors = dict(zip(input_labels,args))

    # convert abcd to half precision and "view as real" form
    tensors['abcd'] = torch.view_as_real(tensors['abcd'])
    tensors['abcd'] = tensors['abcd'].half()

    for partial_eq in partial_eqns:
        # get the input tensors to partial_eq
        in_labels, out_label = partial_eq.split('->')
        in_labels = in_labels.split(',')
        in_tensors = [tensors[label] for label in in_labels]

        if partial_eq in ['ce,e->ce', 'de,be->deb', 'ce,fe->cef']:
            # these are the einsums that work with the complexhalf dtype
            in_tensors[0], in_tensors[1] = in_tensors[0].chalf(), in_tensors[1].chalf()
            tmp = tl.einsum(partial_eq, *in_tensors)
            if partial_eq == 'ce,e->ce':
                # the next einsum will run in complexhalf dtype
                tensors[out_label] = tmp
            else:
                # the next equation will run in half dtype
                tensors[out_label] = torch.view_as_real(tmp)
        else:
            # these are the einsums that do not work with the complexhalf dtype
            #cef,deb->cfdb and cfdb,abcd->afcd

            # create new einsum equation that takes into account "view as real" form
            input_output = partial_eq.split('->')
            new_output = 'xy' + input_output[1]
            input_terms = input_output[0].split(',')
            new_inputs = [input_terms[0] + 'x', input_terms[1] + 'y']
            new_eqn = new_inputs[0] + ',' + new_inputs[1] + '->' + new_output

            # perform the einsum, and convert to "view as real" form
            tmp = tl.einsum(new_eqn, *in_tensors)
            result = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1)
            tensors[out_label] = result

    return torch.view_as_complex(tensors[output_label])