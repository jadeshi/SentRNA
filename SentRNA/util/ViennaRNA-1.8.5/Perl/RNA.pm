# This file was automatically generated by SWIG (http://www.swig.org).
# Version 1.3.39
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

package RNA;
use base qw(Exporter);
use base qw(DynaLoader);
package RNAc;
bootstrap RNA;
package RNA;
@EXPORT = qw();
=head1 NAME

RNA - interface to the Vienna RNA library (libRNA.a)
Version 0.3

=head1 SYNOPSIS

  use RNA;
  $seq = "CGCAGGGAUACCCGCG";
  ($struct, $mfe) = RNA::fold($seq);  #predict mfe structure of $seq
  RNA::PS_rna_plot($seq, $struct, "rna.ps");  # write PS plot to rna.ps
  $F = RNA::pf_fold($seq);   # compute partition function and pair pobabilities
  RNA::PS_dot_plot($seq, "dot.ps");          # write dot plot to dot.ps
  ...

=head1 DESCRIPTION

The RNA.pm package gives access to almost all functions in the libRNA.a
library of the Vienna RNA PACKAGE. The Perl wrapper is generated using
SWIG http://www.swig.org/ with relatively little manual intervention.
For each C function in the library the perl package provides a function
of the same name and calling convention (with few exceptions). For
detailed information you should therefore also consult the documentation
of the library (info RNAlib).

Note that in general C arrays are wrapped into opaque objects that can
only be accessed via helper functions. SWIG provides a couple of general
purpose helper functions, see the section at the end of this file. C
structures are wrapped into Perl objects using SWIG's shadow class
mechanism, resulting in a tied hash with keys named after the structure
members.

For the interrested reader we list for each scalar type of the
corepsonding C variable in brackets, and point out the header files
containing the C declaration.

=head2 Folding Routines

Minimum free Energy Folding (from fold.h)

=over 4

=item fold SEQUENCE

=item fold SEQUENCE, CONSTRAINTS

computes the minimum free energy structure of the string SEQUENCE and returns
the predicted structure and energy, e.g.

  ($structure, $mfe) = RNA::fold("UGUGUCGAUGUGCUAU");

If a second argument is supplied and
L<$fold_constrained|/$fold_constrained>==1 the CONSTRAINTS string is
used to specify constraints on the predicted structure.  The
characters '|', 'x', '<', '>' mark bases that are paired, unpaired,
paired upstream, or downstream, respectively; matching brackets "( )"
denote base pairs, dots '.' are used for unconstrained bases.

In the two argument version the CONSTRAINTS string is modified and holds the
predicted structure upon return. This is done for backwards compatibility only,
and might change in future versions.

=item energy_of_struct SEQUENCE, STRUCTURE

returns the energy of SEQUENCE on STRUCTURE (in kcal/mol). The string structure
must hold a valid secondary structure in bracket notation.

=item update_fold_params

recalculate the pair matrix and energy parameters after a change in folding
parameters. In many cases (such as changes to
L<$temperature|/$temperature>) the fold() routine will call
update_fold_params automatically when necessary.

=item free_arrays

frees memory allocated internally when calling L<fold|/fold>.


=item cofold SEQUENCE

=item cofold SEQUENCE, CONSTRAINTS

works as fold, but SEQUENCE may be the concatenation of two RNAs in order
compute their hybridization structure. E.g.:

  $seq1  ="CGCAGGGAUACCCGCG";
  $seq2  ="GCGCCCAUAGGGACGC";
  $RNA::cut_point = length($seq1)+1;
  ($costruct, $comfe) = RNA::cofold($seq1 . $seq2);

=item duplexfold SEQ1 SEQ2

compute the structure upon hybridization of SEQ1 and SEQ2. In contrast to
cofold only intra-molecular pairs are allowed. Thus, the algorithm runs in
O(n1*n2) time where n1 and n2 are the lengths of the sequences. The result
is returned in a C struct containing the innermost base pair (i,j) the
structure and energy. E.g:

  $seq1 ="CGCAGGGAUACCCGCG";
  $seq2 ="GCGCCCAUAGGGACGC";
  $dup  = RNA::duplexfold($seq1, $seq2);
  print "Region ", $dup->{i}+1-length($seq1), " to ",
	$dup->{i}, " of seq1 ",
	"pairs up with ", $dup->{j}, " to ",
	$dup->{j}+length($dup->{structure})-length($seq1)-2,
	" of seq2\n";

=back

Partition function Folding (from part_func.h)

=over 4

=item pf_fold SEQUENCE

=item pf_fold SEQUENCE, CONSTRAINTS

calculates the partition function over all possible secondary
structures and the matrix of pair probabilities for SEQUENCE and
returns a two element list consisting of a string summarizing possible
structures. See below on how to access the pair probability matrix. As
with L<fold|/fold> the second argument can be used to specify folding
constraints. Constraints are implemented by excluding base pairings
that contradict the constraint, but without bonus
energies. Constraints of type '|' (paired base) are ignored.  In the
two argument version CONSTRAINTS is modified to contain the structure
string on return (obsolete feature, for backwards compatibility only)

=item get_pr I, J

After calling C<pf_fold> the global C variable C<pr> points to the
computed pair probabilities. Perl access to the C is facilitated by
the C<get_pr> helper function that looks up and returns the
probability of the pair (I,J).

=item free_pf_arrays

frees memory allocated for pf_fold

=item update_pf_params LENGTH

recalculate energy parameters for pf_fold. In most cases (such as
simple changes to L<$temperature|/$temperature>) C<pf_fold>
will take appropriate action automatically.



=item pbacktrack SEQUENCE

return a random structure chosen according to it's Boltzmann probability.
Use to produce samples representing the thermodynamic ensemble of
structures.

  RNA::pf_fold($sequence);
  for (1..1000) {
     push @sample, RNA::pbacktrack($sequence);
  }

=item co_pf_fold SEQUENCE

=item co_pf_fold SEQUENCE, CONSTRAINTS

calculates the partition function over all possible secondary
structures and the matrix of pair probabilities for SEQUENCE.
SEQUENCE is a concatenation of two sequences (see cofold).
Returns a five element list consisting of a string summarizing possible
structures as first element. The second element is the Gibbs free energy of Sequence 1 (as computed also with pf_fold), the third element the Gibbs free energy of Sequence 2. The fourth element is the Gibbs free energy of all structures that have INTERmolecular base pairs, and finally the fifth element is the Gibbs free energy of the whole ensemble (dimers as well as monomers).
See above on how to access the pair probability matrix. As
with L<fold|/fold> the second argument can be used to specify folding
constraints. Constraints are implemented by excluding base pairings
that contradict the constraint, but without bonus
energies. Constraints of type '|' (paired base) are ignored.  In the
two argument version CONSTRAINTS is modified to contain the structure
string on return (obsolete feature, for backwards compatibility only)

=item free_co_pf_arrays

frees memory allocated for co_pf_fold

=item update_pf_co_params LENGTH

recalculate energy parameters for co_pf_fold. In most cases (such as
simple changes to L<$temperature|/$temperature>) C<co_pf_fold>
will take appropriate action automatically.

=item get_concentrations FdAB, FdAA, FdBB, FA, FB, CONCA, CONCB

calculates equilibrium concentrations of the three dimers AB, AA, and BB, as well as the two monomers A and B out of the free energies of the duplexes (FdAB, FdAA, FdBB, these are the fourth elements returned by co_pf_fold), the monomers (FA, FB (e.g. the second and third elements returned by co_pf_fold with sequences AB) and the start concentrations of A and B. It returns as first element the concentration of AB dimer, than AA and BB dimer, as fourth element the A monomer concentration, and as fifth and last element the B monomer concentration.
So, to compute concentrations, you first have to run 3 co_pf_folds (with sequences AB, AA and BB).

=back

Suboptimal Folding (from subopt.h)

=over 4

=item subopt SEQUENCE, CONSTRAINTS, DELTA

=item subopt SEQUENCE, CONSTRAINTS, DELTA, FILEHANDLE

compute all structures of SEQUENCE within DELTA*0.01 kcal/mol of the
optimum. If specified, results are written to FILEHANDLE and nothing
is returned. Else, the C function returnes a list of C structs of type
SOLUTION. The list is wrapped by SWIG as a perl object that can be
accesses as follows:

  $solution = subopt($seq, undef, 500);
  for (0..$solution->size()-1) {
     printf "%s %6.2f\n",  $solution->get($_)->{structure},
			   $solution->get($_)->{energy};
  }

=back

Alignment Folding (from alifold.h)

=over 4

=item alifold REF

=item fold REF, CONSTRAINTS

similar to fold() but compute the consensus structure for a set of aligned
sequences. E.g.:

  @align = ("GCCAUCCGAGGGAAAGGUU",
	    "GAUCGACAGCGUCU-AUCG",
	    "CCGUCUUUAUGAGUCCGGC");
  ($consens_struct, $consens_en) = RNA::alifold(\@align);

=item consensus REF
=item consens_mis REF

compute a simple consensus sequence or "most informative sequence" form an
alignment. The simple consensus returns the most frequent character for
each column, the MIS uses the IUPAC symbol that contains all characters
that are overrepresented in the column.

  $mis = consensus_mis(\@align);


=back

Inverse Folding (from inverse.h)

=over 4

=item inverse_fold START, TARGET

find a sequence that folds into structure TARGET, by optimizing the
sequence until its mfe structure (as returned by L<fold|/fold>) is
TARGET. Startpoint of the optimization is the sequence START. Returns
a list containing the sequence found and the final value of the cost
function, i.e. 0 if the search was successful. A random start sequence
can be generated using L<random_string|/random_string>.

=item inverse_pf_fold START, TARGET

optimizes a sequence (beginning with START) by maximising the
frequency of the structure TARGET in the thermodynamic ensemble
of structures. Returns a list containing the optimized sequence and
the final value of the cost function. The cost function is given by
C<energy_of_struct(seq, TARGET) - pf_fold(seq)>, i.e.C<-RT*log(p(TARGET))>

=item $final_cost [float]

holds the value of the cost function where the optimization in
C<inverse_pf_fold> should stop. For values <=0 the optimization will
only terminate at a local optimimum (which might take very long to reach).

=item $symbolset [char *]

the string symbolset holds the allowed characters to be used by
C<inverse_fold> and C<inverse_pf_fold>, the default alphabet is "AUGC"


=item $give_up [int]

If non-zero stop optimization when its clear that no exact solution
can be found. Else continue and eventually return an approximate
solution. Default 0.

=back

Cofolding of two RNA molecules (from cofold.h)

=over 4


=back

Global Variables to Modify Folding (from fold_vars.h)

=over 4

=item $noGU [int]

Do not allow GU pairs to form, default 0.

=item $no_closingGU [int]

allow GU only inside stacks, default 0.

=item $tetra_loop [int]

Fold with specially stable 4-loops, default 1.

=item $energy_set [int]

0 = BP; 1=any mit GC; 2=any mit AU-parameter, default 0.

=item $dangles [int]

How to compute dangling ends. 0: no dangling end energies, 1: "normal"
dangling ends (default), 2: simplified dangling ends, 3: "normal" +
co-axial stacking. Note that L<pf_fold|/pf_fold> treats cases 1 and 3
as 2. The same holds for the main computation in L<subopt|/subopt>,
however subopt will re-evalute energies using
L<energy_of_struct|energy_of_struct> for cases 1 and 3. See the more
detailed discussion in RNAlib.texinfo.

=item $nonstandards [char *]

contains allowed non standard bases, default empty string ""

=item $temperature [double]

temperature in degrees Celsius for rescaling parameters, default 37C.

=item $logML [int]

use logarithmic multiloop energy function in
L<energy_of_struct|/energy_of_struct>, default 0.

=item $noLonelyPairs [int]

consider only structures without isolated base pairs (helices of length 1).
For L<pf_fold|/pf_fold> only eliminates pairs
that can B<only> occur as isolated pairs. Default 0.

=item $base_pair [struct bond *]

list of base pairs from last call to L<fold|/fold>. Better use
the structure string returned by  L<fold|/fold>.

=item $pf_scale [double]

scaling factor used by L<pf_fold|/pf_fold> to avoid overflows. Should
be set to exp(-F/(RT*length)) where F is a guess for the ensmble free
energy (e.g. use the mfe).


=item $fold_constrained [int]

apply constraints in the folding algorithms, default 0.

=item $do_backtrack [int]

If 0 do not compute the pair probabilities in L<pf_fold|/pf_fold>
(only the partition function). Default 1.

=item $backtrack_type [char]

usually 'F'; 'C' require (1,N) to be bonded; 'M' backtrack as if the
sequence was part of a multi loop. Used by L<inverse_fold|/inverse_fold>

=item $pr [double *]

the base pairing prob. matrix computed by L<pf_fold|/pf_fold>.

=item $iindx [int *]

Array of indices for moving withing the C<pr> array. Better use
L<get_pr|/get_pr>.


=back

=head2 Parsing and Comparing Structures

from RNAstruct.h: these functions convert between strings
representating secondary structures with various levels of coarse
graining. See the documentation of the C library for details

=over 4

=item b2HIT STRUCTURE

Full -> HIT [incl. root]

=item b2C STRUCTURE

Full -> Coarse [incl. root]

=item b2Shapiro STRUCTURE

Full -> weighted Shapiro [i.r.]

=item add_root STRUCTURE

{Tree} -> ({Tree}R)

=item expand_Shapiro COARSE

add S for stacks to coarse struct

=item expand_Full STRUCTURE

Full -> FFull

=item unexpand_Full FSTRUCTURE

FFull -> Full

=item unweight WCOARSE

remove weights from coarse struct

=item unexpand_aligned_F ALIGN



=item parse_structure STRUCTURE

computes structure statistics, and fills the following global variables:

$loops    [int] number of loops (and stacks)
$unpaired [int] number of unpaired positions
$pairs    [int] number of paired positions
$loop_size[int *]  holds all loop sizes
$loop_degree[int *] holds all loop degrees
$helix_size[int *] holds all helix lengths

=back

from treedist.h: routines for computing tree-edit distances between structures

=over 4

=item make_tree XSTRUCT

convert a structure string as produced by the expand_... functions to a
Tree, useable as input to tree_edit_distance.

=item tree_edit_distance T1, T2

compare to structures using tree editing. C<T1>, C<T2> must have been
created using C<tree_edit_distance>

=item print_tree T

mainly for debugging

=item free_tree T

free space allocated by make_tree

=back

from stringdist.h routines to compute structure distances via string-editing

=over 4

=item Make_swString STRUCTURE

[ returns swString * ]
make input for string_edit_distance

=item string_edit_distance S1, S2

[ returns float  ]
compare to structures using string alignment. C<S1>, C<S2> should be
created using C<Make_swString>

=back

from profiledist

=over

=item Make_bp_profile LENGTH

[ returns (float *) ]
condense pair probability matrix C<pr> into a vector containing
probabilities for upstream paired, downstream paired and
unpaired. This resulting probability profile is used as input for
profile_edit_distance

=item profile_edit_distance T1, T2

[ returns float ]
align two probability profiles produced by C<Make_bp_profile>

=item print_bppm T

[ returns void ]
print string representation of probability profile

=item free_profile T

[ returns void ]
free space allocated in Make_bp_profile

=back

Global variables for computing structure distances

=over 4

=item $edit_backtrack [int]

set to 1 if you want backtracking

=item $aligned_line [(char *)[2]]

containes alignmed structures after computing structure distance with
C<edit_backtrack==1>

=item $cost_matrix [int]

0 usual costs (default), 1 Shapiro's costs

=back

=head2 Utilities (from utils.h)

=over 4

=item space SIZE

allocate memory from C. Usually not needed in Perl

=item nrerror MESSGAE

die with error message. Better use Perl's C<die>

=item $xsubi [unsigned short[3]]

libRNA uses the rand48 48bit random number generator if available, the
current random  number is always stored in $xsubi.

=item init_rand

initialize the $xsubi random number from current time

=item urn

returns a random number between 0 and 1 using the random number
generator from the RNA library.

=item int_urn FROM, TO

returns random integer in the range [FROM..TO]

=item time_stamp

current date in a string. In perl you might as well use C<locatime>

=item random_string LENGTH, SYMBOLS

returns a string of length LENGTH using characters from the string
SYMBOLS

=item hamming S1, S2

calculate hamming distance of the strings C<S1> and C<S2>.


=item pack_structure STRUCTURE

pack secondary structure, using a 5:1 compression via 3
encoding. Returns the packed string.

=item unpack_structure PACKED

unpacks a secondary structure packed with pack_structure

=item make_pair_table STRUCTURE

returns a pair table as a newly allocated (short *) C array, such
that: table[i]=j if (i.j) pair or 0 if i is unpaired, table[0]
contains the length of the structure.

=item bp_distance STRUCTURE1, STRUCTURE2

returns the base pair distance of the two STRUCTURES. dist = {number
of base pairs in one structure but not in the other} same as edit
distance with open-pair close-pair as move-set

=back

from PS_plot.h

=over 4

=item PS_rna_plot SEQUENCE, STRUCTURE, FILENAME

write PostScript drawing of structure to FILENAME. Returns 1 on
sucess, 0 else.

=item PS_rna_plot_a SEQUENCE, STRUCTURE, FILENAME, PRE, POST

write PostScript drawing of structure to FILENAME. The strings PRE and
POST contain PostScript code that is included verbatim in the plot just
before (after) the data.  Returns 1 on sucess, 0 else.

=item gmlRNA SEQUENCE, STRUCTURE, FILENAME, OPTION

write structure drawing in gml (Graph Meta Language) to
FILENAME. OPTION should be a single character. If uppercase the gml
output will include the SEQUENCE as node labels. IF OPTION equal 'x'
or 'X' write graph with coordinates (else only connectivity
information). Returns 1 on sucess, 0 else.

=item ssv_rna_plot SEQUENCE, STRUCTURE, SSFILE

write structure drfawing as coord file for SStructView Returns 1 on
sucess, 0 else.

=item xrna_plot SEQUENCE, STRUCTURE, SSFILE

write structure drawing as ".ss" file for further editing in XRNA.
Returns 1 on sucess, 0 else.

=item PS_dot_plot SEQUENCE, FILENAME

write a PostScript dot plot of the pair probability matix to
FILENAME. Returns 1 on sucess, 0 else.

=item $rna_plot_type [int]

Select layout algorithm for structure drawings. Currently available
0= simple coordinates, 1= naview, default 1.

=back

from read_epars.c

=over 4

=item read_parameter_file FILENAME

read energy parameters from FILENAME

=item write_parameter_file FILENAME

write energy parameters to FILENAME

=back

=head2 SWIG helper functions

The package includes generic helper functions to access C arrays
of type C<int>, C<float> and C<double>, such as:

=over 4

=item intP_getitem POINTER, INDEX

return the element INDEX from the array

=item intP_setitem POINTER, INDEX, VALUE

set element INDEX to VALUE

=item new_intP NELEM

allocate a new C array of integers with NELEM elements and return the pointer

=item delete_intP POINTER

deletes the C array by calling free()

=back

substituting C<intP> with C<floatP>, C<doubleP>, C<ushortP>,
C<shortP>, gives the corresponding functions for arrays of float or
double, unsigned short, and short. You need to know the correct C
type however, and the functions work only for arrays of simple types.
Note, that the shortP... functions were used for unsigned short in previous 
versions, while starting with v1.8.3 it can only access signed short arrays. 

On the lowest level the C<cdata> function gives direct access to any data
in the form of a Perl string.

=over

=item cdata POINTER, SIZE

copies SIZE bytes at POINTER to a Perl string (with binary data)

=item memmove POINTER, STRING

copies the (binary) string STRING to the memory location pointed to by
POINTER.
Note: memmove is broken in current swig versions (e.g. 1.3.31)

=back

In combination with Perl's C<unpack> this provides a generic way to convert
C data structures to Perl. E.g.

  RNA::parse_structure($structure);  # fills the $RNA::loop_degree array
  @ldegrees = unpack "I*", RNA::cdata($RNA::loop_degree, ($RNA::loops+1)*4);

Warning: using these functions with wrong arguments will corrupt your
memory and lead to a segmentation fault.

=head1 AUTHOR

Ivo L. Hofacker <ivo@tbi.univie.ac.at>

=cut

# ---------- BASE METHODS -------------

package RNA;

sub TIEHASH {
    my ($classname,$obj) = @_;
    return bless $obj, $classname;
}

sub CLEAR { }

sub FIRSTKEY { }

sub NEXTKEY { }

sub FETCH {
    my ($self,$field) = @_;
    my $member_func = "swig_${field}_get";
    $self->$member_func();
}

sub STORE {
    my ($self,$field,$newval) = @_;
    my $member_func = "swig_${field}_set";
    $self->$member_func($newval);
}

sub this {
    my $ptr = shift;
    return tied(%$ptr);
}


# ------- FUNCTION WRAPPERS --------

package RNA;

*new_intP = *RNAc::new_intP;
*delete_intP = *RNAc::delete_intP;
*intP_getitem = *RNAc::intP_getitem;
*intP_setitem = *RNAc::intP_setitem;
*new_floatP = *RNAc::new_floatP;
*delete_floatP = *RNAc::delete_floatP;
*floatP_getitem = *RNAc::floatP_getitem;
*floatP_setitem = *RNAc::floatP_setitem;
*new_doubleP = *RNAc::new_doubleP;
*delete_doubleP = *RNAc::delete_doubleP;
*doubleP_getitem = *RNAc::doubleP_getitem;
*doubleP_setitem = *RNAc::doubleP_setitem;
*new_ushortP = *RNAc::new_ushortP;
*delete_ushortP = *RNAc::delete_ushortP;
*ushortP_getitem = *RNAc::ushortP_getitem;
*ushortP_setitem = *RNAc::ushortP_setitem;
*new_shortP = *RNAc::new_shortP;
*delete_shortP = *RNAc::delete_shortP;
*shortP_getitem = *RNAc::shortP_getitem;
*shortP_setitem = *RNAc::shortP_setitem;
*cdata = *RNAc::cdata;
*memmove = *RNAc::memmove;
*fold = *RNAc::fold;
*energy_of_struct = *RNAc::energy_of_struct;
*free_arrays = *RNAc::free_arrays;
*initialize_fold = *RNAc::initialize_fold;
*update_fold_params = *RNAc::update_fold_params;
*backtrack_fold_from_pair = *RNAc::backtrack_fold_from_pair;
*loop_energy = *RNAc::loop_energy;
*export_fold_arrays = *RNAc::export_fold_arrays;
*circfold = *RNAc::circfold;
*energy_of_circ_struct = *RNAc::energy_of_circ_struct;
*export_circfold_arrays = *RNAc::export_circfold_arrays;
*cofold = *RNAc::cofold;
*free_co_arrays = *RNAc::free_co_arrays;
*initialize_cofold = *RNAc::initialize_cofold;
*update_cofold_params = *RNAc::update_cofold_params;
*zukersubopt = *RNAc::zukersubopt;
*pf_fold = *RNAc::pf_fold;
*init_pf_fold = *RNAc::init_pf_fold;
*free_pf_arrays = *RNAc::free_pf_arrays;
*update_pf_params = *RNAc::update_pf_params;
*bppm_symbol = *RNAc::bppm_symbol;
*mean_bp_dist = *RNAc::mean_bp_dist;
*centroid = *RNAc::centroid;
*get_pf_arrays = *RNAc::get_pf_arrays;
*pbacktrack = *RNAc::pbacktrack;
*pf_circ_fold = *RNAc::pf_circ_fold;
*pbacktrack_circ = *RNAc::pbacktrack_circ;
*co_pf_fold = *RNAc::co_pf_fold;
*free_co_pf_arrays = *RNAc::free_co_pf_arrays;
*update_co_pf_params = *RNAc::update_co_pf_params;
*get_concentrations = *RNAc::get_concentrations;
*inverse_fold = *RNAc::inverse_fold;
*inverse_pf_fold = *RNAc::inverse_pf_fold;
*option_string = *RNAc::option_string;
*free_alifold_arrays = *RNAc::free_alifold_arrays;
*alipf_fold = *RNAc::alipf_fold;
*centroid_ali = *RNAc::centroid_ali;
*readribosum = *RNAc::readribosum;
*alipbacktrack = *RNAc::alipbacktrack;
*free_alipf_arrays = *RNAc::free_alipf_arrays;
*energy_of_alistruct = *RNAc::energy_of_alistruct;
*circalifold = *RNAc::circalifold;
*alipf_circ_fold = *RNAc::alipf_circ_fold;
*alifold = *RNAc::alifold;
*consensus = *RNAc::consensus;
*consens_mis = *RNAc::consens_mis;
*get_xy_coordinates = *RNAc::get_xy_coordinates;
*subopt = *RNAc::subopt;
*get_pr = *RNAc::get_pr;
*b2HIT = *RNAc::b2HIT;
*b2C = *RNAc::b2C;
*b2Shapiro = *RNAc::b2Shapiro;
*add_root = *RNAc::add_root;
*expand_Shapiro = *RNAc::expand_Shapiro;
*expand_Full = *RNAc::expand_Full;
*unexpand_Full = *RNAc::unexpand_Full;
*unweight = *RNAc::unweight;
*unexpand_aligned_F = *RNAc::unexpand_aligned_F;
*parse_structure = *RNAc::parse_structure;
*make_tree = *RNAc::make_tree;
*tree_edit_distance = *RNAc::tree_edit_distance;
*print_tree = *RNAc::print_tree;
*free_tree = *RNAc::free_tree;
*Make_swString = *RNAc::Make_swString;
*string_edit_distance = *RNAc::string_edit_distance;
*Make_bp_profile = *RNAc::Make_bp_profile;
*profile_edit_distance = *RNAc::profile_edit_distance;
*print_bppm = *RNAc::print_bppm;
*free_profile = *RNAc::free_profile;
*space = *RNAc::space;
*xrealloc = *RNAc::xrealloc;
*nrerror = *RNAc::nrerror;
*init_rand = *RNAc::init_rand;
*urn = *RNAc::urn;
*int_urn = *RNAc::int_urn;
*filecopy = *RNAc::filecopy;
*time_stamp = *RNAc::time_stamp;
*random_string = *RNAc::random_string;
*hamming = *RNAc::hamming;
*get_line = *RNAc::get_line;
*pack_structure = *RNAc::pack_structure;
*unpack_structure = *RNAc::unpack_structure;
*make_pair_table = *RNAc::make_pair_table;
*bp_distance = *RNAc::bp_distance;
*read_parameter_file = *RNAc::read_parameter_file;
*write_parameter_file = *RNAc::write_parameter_file;
*deref_any = *RNAc::deref_any;
*scale_parameters = *RNAc::scale_parameters;
*copy_parameters = *RNAc::copy_parameters;
*set_parameters = *RNAc::set_parameters;
*get_aligned_line = *RNAc::get_aligned_line;
*make_loop_index = *RNAc::make_loop_index;
*energy_of_move = *RNAc::energy_of_move;
*duplexfold = *RNAc::duplexfold;
*aliduplexfold = *RNAc::aliduplexfold;
*encode_seq = *RNAc::encode_seq;
*Lfold = *RNAc::Lfold;
*PS_rna_plot = *RNAc::PS_rna_plot;
*PS_rna_plot_a = *RNAc::PS_rna_plot_a;
*gmlRNA = *RNAc::gmlRNA;
*ssv_rna_plot = *RNAc::ssv_rna_plot;
*svg_rna_plot = *RNAc::svg_rna_plot;
*xrna_plot = *RNAc::xrna_plot;
*PS_dot_plot = *RNAc::PS_dot_plot;
*PS_color_dot_plot = *RNAc::PS_color_dot_plot;
*PS_color_dot_plot_turn = *RNAc::PS_color_dot_plot_turn;
*PS_dot_plot_list = *RNAc::PS_dot_plot_list;
*PS_dot_plot_turn = *RNAc::PS_dot_plot_turn;
*PS_color_aln = *RNAc::PS_color_aln;
*find_saddle = *RNAc::find_saddle;
*get_path = *RNAc::get_path;

############# Class : RNA::intArray ##############

package RNA::intArray;
use vars qw(@ISA %OWNER %ITERATORS %BLESSEDMEMBERS);
@ISA = qw( RNA );
%OWNER = ();
%ITERATORS = ();
sub new {
    my $pkg = shift;
    my $self = RNAc::new_intArray(@_);
    bless $self, $pkg if defined($self);
}

sub DESTROY {
    return unless $_[0]->isa('HASH');
    my $self = tied(%{$_[0]});
    return unless defined $self;
    delete $ITERATORS{$self};
    if (exists $OWNER{$self}) {
        RNAc::delete_intArray($self);
        delete $OWNER{$self};
    }
}

*getitem = *RNAc::intArray_getitem;
*setitem = *RNAc::intArray_setitem;
*cast = *RNAc::intArray_cast;
*frompointer = *RNAc::intArray_frompointer;
sub DISOWN {
    my $self = shift;
    my $ptr = tied(%$self);
    delete $OWNER{$ptr};
}

sub ACQUIRE {
    my $self = shift;
    my $ptr = tied(%$self);
    $OWNER{$ptr} = 1;
}


############# Class : RNA::floatArray ##############

package RNA::floatArray;
use vars qw(@ISA %OWNER %ITERATORS %BLESSEDMEMBERS);
@ISA = qw( RNA );
%OWNER = ();
%ITERATORS = ();
sub new {
    my $pkg = shift;
    my $self = RNAc::new_floatArray(@_);
    bless $self, $pkg if defined($self);
}

sub DESTROY {
    return unless $_[0]->isa('HASH');
    my $self = tied(%{$_[0]});
    return unless defined $self;
    delete $ITERATORS{$self};
    if (exists $OWNER{$self}) {
        RNAc::delete_floatArray($self);
        delete $OWNER{$self};
    }
}

*getitem = *RNAc::floatArray_getitem;
*setitem = *RNAc::floatArray_setitem;
*cast = *RNAc::floatArray_cast;
*frompointer = *RNAc::floatArray_frompointer;
sub DISOWN {
    my $self = shift;
    my $ptr = tied(%$self);
    delete $OWNER{$ptr};
}

sub ACQUIRE {
    my $self = shift;
    my $ptr = tied(%$self);
    $OWNER{$ptr} = 1;
}


############# Class : RNA::doubleArray ##############

package RNA::doubleArray;
use vars qw(@ISA %OWNER %ITERATORS %BLESSEDMEMBERS);
@ISA = qw( RNA );
%OWNER = ();
%ITERATORS = ();
sub new {
    my $pkg = shift;
    my $self = RNAc::new_doubleArray(@_);
    bless $self, $pkg if defined($self);
}

sub DESTROY {
    return unless $_[0]->isa('HASH');
    my $self = tied(%{$_[0]});
    return unless defined $self;
    delete $ITERATORS{$self};
    if (exists $OWNER{$self}) {
        RNAc::delete_doubleArray($self);
        delete $OWNER{$self};
    }
}

*getitem = *RNAc::doubleArray_getitem;
*setitem = *RNAc::doubleArray_setitem;
*cast = *RNAc::doubleArray_cast;
*frompointer = *RNAc::doubleArray_frompointer;
sub DISOWN {
    my $self = shift;
    my $ptr = tied(%$self);
    delete $OWNER{$ptr};
}

sub ACQUIRE {
    my $self = shift;
    my $ptr = tied(%$self);
    $OWNER{$ptr} = 1;
}


############# Class : RNA::bondT ##############

package RNA::bondT;
use vars qw(@ISA %OWNER %ITERATORS %BLESSEDMEMBERS);
@ISA = qw( RNA );
%OWNER = ();
%ITERATORS = ();
*swig_i_get = *RNAc::bondT_i_get;
*swig_i_set = *RNAc::bondT_i_set;
*swig_j_get = *RNAc::bondT_j_get;
*swig_j_set = *RNAc::bondT_j_set;
*get = *RNAc::bondT_get;
sub new {
    my $pkg = shift;
    my $self = RNAc::new_bondT(@_);
    bless $self, $pkg if defined($self);
}

sub DESTROY {
    return unless $_[0]->isa('HASH');
    my $self = tied(%{$_[0]});
    return unless defined $self;
    delete $ITERATORS{$self};
    if (exists $OWNER{$self}) {
        RNAc::delete_bondT($self);
        delete $OWNER{$self};
    }
}

sub DISOWN {
    my $self = shift;
    my $ptr = tied(%$self);
    delete $OWNER{$ptr};
}

sub ACQUIRE {
    my $self = shift;
    my $ptr = tied(%$self);
    $OWNER{$ptr} = 1;
}


############# Class : RNA::pair_info ##############

package RNA::pair_info;
use vars qw(@ISA %OWNER %ITERATORS %BLESSEDMEMBERS);
@ISA = qw( RNA );
%OWNER = ();
%ITERATORS = ();
*swig_i_get = *RNAc::pair_info_i_get;
*swig_i_set = *RNAc::pair_info_i_set;
*swig_j_get = *RNAc::pair_info_j_get;
*swig_j_set = *RNAc::pair_info_j_set;
*swig_p_get = *RNAc::pair_info_p_get;
*swig_p_set = *RNAc::pair_info_p_set;
*swig_ent_get = *RNAc::pair_info_ent_get;
*swig_ent_set = *RNAc::pair_info_ent_set;
*swig_bp_get = *RNAc::pair_info_bp_get;
*swig_bp_set = *RNAc::pair_info_bp_set;
*swig_comp_get = *RNAc::pair_info_comp_get;
*swig_comp_set = *RNAc::pair_info_comp_set;
sub new {
    my $pkg = shift;
    my $self = RNAc::new_pair_info(@_);
    bless $self, $pkg if defined($self);
}

sub DESTROY {
    return unless $_[0]->isa('HASH');
    my $self = tied(%{$_[0]});
    return unless defined $self;
    delete $ITERATORS{$self};
    if (exists $OWNER{$self}) {
        RNAc::delete_pair_info($self);
        delete $OWNER{$self};
    }
}

sub DISOWN {
    my $self = shift;
    my $ptr = tied(%$self);
    delete $OWNER{$ptr};
}

sub ACQUIRE {
    my $self = shift;
    my $ptr = tied(%$self);
    $OWNER{$ptr} = 1;
}


############# Class : RNA::COORDINATE ##############

package RNA::COORDINATE;
use vars qw(@ISA %OWNER %ITERATORS %BLESSEDMEMBERS);
@ISA = qw( RNA );
%OWNER = ();
%ITERATORS = ();
*swig_X_get = *RNAc::COORDINATE_X_get;
*swig_X_set = *RNAc::COORDINATE_X_set;
*swig_Y_get = *RNAc::COORDINATE_Y_get;
*swig_Y_set = *RNAc::COORDINATE_Y_set;
*get = *RNAc::COORDINATE_get;
sub new {
    my $pkg = shift;
    my $self = RNAc::new_COORDINATE(@_);
    bless $self, $pkg if defined($self);
}

sub DESTROY {
    return unless $_[0]->isa('HASH');
    my $self = tied(%{$_[0]});
    return unless defined $self;
    delete $ITERATORS{$self};
    if (exists $OWNER{$self}) {
        RNAc::delete_COORDINATE($self);
        delete $OWNER{$self};
    }
}

sub DISOWN {
    my $self = shift;
    my $ptr = tied(%$self);
    delete $OWNER{$ptr};
}

sub ACQUIRE {
    my $self = shift;
    my $ptr = tied(%$self);
    $OWNER{$ptr} = 1;
}


############# Class : RNA::SOLUTION ##############

package RNA::SOLUTION;
use vars qw(@ISA %OWNER %ITERATORS %BLESSEDMEMBERS);
@ISA = qw( RNA );
%OWNER = ();
%ITERATORS = ();
*swig_energy_get = *RNAc::SOLUTION_energy_get;
*swig_energy_set = *RNAc::SOLUTION_energy_set;
*swig_structure_get = *RNAc::SOLUTION_structure_get;
*swig_structure_set = *RNAc::SOLUTION_structure_set;
*get = *RNAc::SOLUTION_get;
*size = *RNAc::SOLUTION_size;
sub DESTROY {
    return unless $_[0]->isa('HASH');
    my $self = tied(%{$_[0]});
    return unless defined $self;
    delete $ITERATORS{$self};
    if (exists $OWNER{$self}) {
        RNAc::delete_SOLUTION($self);
        delete $OWNER{$self};
    }
}

sub new {
    my $pkg = shift;
    my $self = RNAc::new_SOLUTION(@_);
    bless $self, $pkg if defined($self);
}

sub DISOWN {
    my $self = shift;
    my $ptr = tied(%$self);
    delete $OWNER{$ptr};
}

sub ACQUIRE {
    my $self = shift;
    my $ptr = tied(%$self);
    $OWNER{$ptr} = 1;
}


############# Class : RNA::duplexT ##############

package RNA::duplexT;
use vars qw(@ISA %OWNER %ITERATORS %BLESSEDMEMBERS);
@ISA = qw( RNA );
%OWNER = ();
%ITERATORS = ();
*swig_i_get = *RNAc::duplexT_i_get;
*swig_i_set = *RNAc::duplexT_i_set;
*swig_j_get = *RNAc::duplexT_j_get;
*swig_j_set = *RNAc::duplexT_j_set;
*swig_structure_get = *RNAc::duplexT_structure_get;
*swig_structure_set = *RNAc::duplexT_structure_set;
*swig_energy_get = *RNAc::duplexT_energy_get;
*swig_energy_set = *RNAc::duplexT_energy_set;
sub new {
    my $pkg = shift;
    my $self = RNAc::new_duplexT(@_);
    bless $self, $pkg if defined($self);
}

sub DESTROY {
    return unless $_[0]->isa('HASH');
    my $self = tied(%{$_[0]});
    return unless defined $self;
    delete $ITERATORS{$self};
    if (exists $OWNER{$self}) {
        RNAc::delete_duplexT($self);
        delete $OWNER{$self};
    }
}

sub DISOWN {
    my $self = shift;
    my $ptr = tied(%$self);
    delete $OWNER{$ptr};
}

sub ACQUIRE {
    my $self = shift;
    my $ptr = tied(%$self);
    $OWNER{$ptr} = 1;
}


############# Class : RNA::cpair ##############

package RNA::cpair;
use vars qw(@ISA %OWNER %ITERATORS %BLESSEDMEMBERS);
@ISA = qw( RNA );
%OWNER = ();
%ITERATORS = ();
*swig_i_get = *RNAc::cpair_i_get;
*swig_i_set = *RNAc::cpair_i_set;
*swig_j_get = *RNAc::cpair_j_get;
*swig_j_set = *RNAc::cpair_j_set;
*swig_mfe_get = *RNAc::cpair_mfe_get;
*swig_mfe_set = *RNAc::cpair_mfe_set;
*swig_p_get = *RNAc::cpair_p_get;
*swig_p_set = *RNAc::cpair_p_set;
*swig_hue_get = *RNAc::cpair_hue_get;
*swig_hue_set = *RNAc::cpair_hue_set;
*swig_sat_get = *RNAc::cpair_sat_get;
*swig_sat_set = *RNAc::cpair_sat_set;
sub new {
    my $pkg = shift;
    my $self = RNAc::new_cpair(@_);
    bless $self, $pkg if defined($self);
}

sub DESTROY {
    return unless $_[0]->isa('HASH');
    my $self = tied(%{$_[0]});
    return unless defined $self;
    delete $ITERATORS{$self};
    if (exists $OWNER{$self}) {
        RNAc::delete_cpair($self);
        delete $OWNER{$self};
    }
}

sub DISOWN {
    my $self = shift;
    my $ptr = tied(%$self);
    delete $OWNER{$ptr};
}

sub ACQUIRE {
    my $self = shift;
    my $ptr = tied(%$self);
    $OWNER{$ptr} = 1;
}


############# Class : RNA::plist ##############

package RNA::plist;
use vars qw(@ISA %OWNER %ITERATORS %BLESSEDMEMBERS);
@ISA = qw( RNA );
%OWNER = ();
%ITERATORS = ();
*swig_i_get = *RNAc::plist_i_get;
*swig_i_set = *RNAc::plist_i_set;
*swig_j_get = *RNAc::plist_j_get;
*swig_j_set = *RNAc::plist_j_set;
*swig_p_get = *RNAc::plist_p_get;
*swig_p_set = *RNAc::plist_p_set;
sub new {
    my $pkg = shift;
    my $self = RNAc::new_plist(@_);
    bless $self, $pkg if defined($self);
}

sub DESTROY {
    return unless $_[0]->isa('HASH');
    my $self = tied(%{$_[0]});
    return unless defined $self;
    delete $ITERATORS{$self};
    if (exists $OWNER{$self}) {
        RNAc::delete_plist($self);
        delete $OWNER{$self};
    }
}

sub DISOWN {
    my $self = shift;
    my $ptr = tied(%$self);
    delete $OWNER{$ptr};
}

sub ACQUIRE {
    my $self = shift;
    my $ptr = tied(%$self);
    $OWNER{$ptr} = 1;
}


############# Class : RNA::path_t ##############

package RNA::path_t;
use vars qw(@ISA %OWNER %ITERATORS %BLESSEDMEMBERS);
@ISA = qw( RNA );
%OWNER = ();
%ITERATORS = ();
*swig_en_get = *RNAc::path_t_en_get;
*swig_en_set = *RNAc::path_t_en_set;
*swig_s_get = *RNAc::path_t_s_get;
*swig_s_set = *RNAc::path_t_s_set;
sub new {
    my $pkg = shift;
    my $self = RNAc::new_path_t(@_);
    bless $self, $pkg if defined($self);
}

sub DESTROY {
    return unless $_[0]->isa('HASH');
    my $self = tied(%{$_[0]});
    return unless defined $self;
    delete $ITERATORS{$self};
    if (exists $OWNER{$self}) {
        RNAc::delete_path_t($self);
        delete $OWNER{$self};
    }
}

sub DISOWN {
    my $self = shift;
    my $ptr = tied(%$self);
    delete $OWNER{$ptr};
}

sub ACQUIRE {
    my $self = shift;
    my $ptr = tied(%$self);
    $OWNER{$ptr} = 1;
}


# ------- VARIABLE STUBS --------

package RNA;

*VERSION = *RNAc::VERSION;
*mirnatog = *RNAc::mirnatog;
*symbolset = *RNAc::symbolset;
*final_cost = *RNAc::final_cost;
*give_up = *RNAc::give_up;
*noGU = *RNAc::noGU;
*no_closingGU = *RNAc::no_closingGU;
*tetra_loop = *RNAc::tetra_loop;
*energy_set = *RNAc::energy_set;
*dangles = *RNAc::dangles;
*oldAliEn = *RNAc::oldAliEn;
*ribo = *RNAc::ribo;
*RibosumFile = *RNAc::RibosumFile;
*nonstandards = *RNAc::nonstandards;
*temperature = *RNAc::temperature;
*james_rule = *RNAc::james_rule;
*logML = *RNAc::logML;
*cut_point = *RNAc::cut_point;

my %__base_pair_hash;
tie %__base_pair_hash,"RNA::bondT", $RNAc::base_pair;
$base_pair= \%__base_pair_hash;
bless $base_pair, RNA::bondT;
*pr = *RNAc::pr;
*iindx = *RNAc::iindx;
*pf_scale = *RNAc::pf_scale;
*fold_constrained = *RNAc::fold_constrained;
*do_backtrack = *RNAc::do_backtrack;
*noLonelyPairs = *RNAc::noLonelyPairs;
*backtrack_type = *RNAc::backtrack_type;
*cv_fact = *RNAc::cv_fact;
*nc_fact = *RNAc::nc_fact;
*subopt_sorted = *RNAc::subopt_sorted;
*loop_size = *RNAc::loop_size;
*helix_size = *RNAc::helix_size;
*loop_degree = *RNAc::loop_degree;
*loops = *RNAc::loops;
*unpaired = *RNAc::unpaired;
*pairs = *RNAc::pairs;
*edit_backtrack = *RNAc::edit_backtrack;
*aligned_line = *RNAc::aligned_line;
*cost_matrix = *RNAc::cost_matrix;
*xsubi = *RNAc::xsubi;
*rna_plot_type = *RNAc::rna_plot_type;
1;
