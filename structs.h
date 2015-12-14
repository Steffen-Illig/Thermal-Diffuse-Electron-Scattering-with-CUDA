/*					*** struct.h ***
              	
This is a reworked application of the kirkland multislice code for calculating 
electron waves from the crystal potential. Please reference:
  
Kirkland, E. J. 'Advanced Computing in Electron Microscopy', Plenum Press,
New York, 1998

------------------------------------------------------------------------
Copyright 1998 Earl J. Kirkland
The computer code and or data in this file is provided for demonstration
purposes only with no guarantee or warranty of any kind that it is correct
or produces correct results.  By using the code and or data in this file
the user agrees to accept all risks and liabilities associated with the code
and or data. The computer code and or data in this file may be copied (and used)
for non-commercial academic or research purposes only, provided that this
notice is included. This file or any portion of it may not be resold, rented
or distributed without the written permission of the author.

The original code has been significantly altered for use under GPU conditions
and for calculating frozen phonons dynamically. Please reference:
  
	Eggeman, A. S et al. Ultramicroscopy, 134 (2013), pp 44-47.
	Illig, S. et al. Nature Communication (2015)
  
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _struct_defs_H_
#define _struct_defs_H_

#include <cufft.h>
#include <string>
#include <vector>	
using namespace std;
 

    struct struct_atom {
       string type;
	   string label;
	   int atomic_no;
       vector<float> pos;
	   int group;
	   struct_atom(): pos(3,0) {}	//vecor length is set to 3
       };          

    struct struct_unitcellprop {
	   float abc[3];			 //in Ang
       float angle[3];           //in degree
	   float volume;			 //In Ang^3 (computed automatically)
	   vector<struct_atom> atoms; 
       };        

	struct struct_coordinatesystem {
		unsigned int definingatoms[3];
		vector< vector<float> > d;
		};

	struct struct_phonons {	
		float sigma;
		int group;
		int coordsystem;
		string type;							//Either "rot" or "trans"
		vector<float> d;
		struct_phonons(): d(3,0) {}				//vecor length is set to 3
		};

	struct struct_supercell {	
		string name;
		unsigned int scsize[3];
		vector<struct_coordinatesystem> sys;
		vector<struct_phonons> ph;
		float sigma_random_noise;
		struct_unitcellprop *cell;
		};

	struct Complex {
		float re;
		float im;
		};

	struct struct_CPU_data {
		vector<Complex> wave;			//Exitwave
		int noofelements;
		vector<int> link;				//Link function. Say we have a materials with 3 elements: C, Si and S. We thus have 3 d_scat array, e.g. d_scat[0] pointing to C, d_scat[1] to Si etc. Then link[6] = 0, link[14] = 1 etc
	};

	struct struct_GPU_data {																	
		Complex *d_trans;				//prefix 'd_' means array on GPU																	
		Complex *d_wave;																	
		Complex *d_prop;
		Complex *d_scat;				
		float *d_kx, *d_ky, *d_k2;		//d_kx, d_ky are vectors with length nx, d_k2 = d_kx^2 + d_ky^2 are nx*ny arrays - reciprocal lattice arrays
	};

	struct struct_parameters {	
		float var1, var2, var3;
		string output;
		string prefix;																//Material Prefix (string to identify input and output files)
		bool thickness_scan;														//Provides intermediate outputs for all thicknesses
		int averaging;																//Runs the simulation averaging times and averages the final output
		float slice_thickness;														//Thickness of each slice in Angstroem (this is adjusted so that the super cell thickness is a multiple)
		int totalslices;															
		float DWF;
		vector<float> hkl;															//Stores the output values
		float tension;																//Acceleration voltage
		float wavelen;																//Resulting wavelength
		float mm0;																	
		int nx, ny;																	//Image dimentsions in px
		float k2max;																//Require to cut off highest frequencies
		cufftHandle plan;															
		struct_supercell *sc;														//Read from input file
		struct_unitcellprop *cif;													//Structural data (read from cif-input file)
		struct_CPU_data *CPU;
		struct_GPU_data *GPU;
	};

#endif
