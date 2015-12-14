/*					*** main.cu ***
              	
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

#include <stdio.h>		// ANSI C libraries, printf()
#include <stdlib.h>		// sleep(), exit(): e.g. exit(1) for failed execution - exits the program
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <cstdlib>
#include <iomanip>      // std::setprecision
#include <string>
#include <fstream>
#include <sstream>
#include <list>			// Lists
#include <vector>		// Better use vector than C-dynamic arrays via malloc
#include <array>
#include <cuda.h>		// CUDA stuff
#include <ctime>
using namespace std;	

#include "structs.h"
#include "functions.h"
#include "init.h"
#include "output.h"

void RunTurboSlice(struct_parameters *par);

int main(int argc, char *argv[]) {

	cout << "Initialize..." << endl;
	srand( (unsigned int) time(NULL) );
	struct_parameters *par = new struct_parameters;
	//Prompt User for Input prefix
	cout << ">> Enter prefix for input data\n";
	cin >> par->prefix;	

	Init_General(par);
	Init_ReadStructureFile(par);
	Init_ReadInputFile(par);
	Init_ProcessInputData(par);			//Does a number of adjustments to bring the input data into the expected form (see function for details)
	for (int a = 0; a < par->averaging; a++) {
		cout << "\n----- Begin Run " << (a + 1) << " of " <<  par->averaging << " -----\n";
		CalculateSuperCell(par);								//Calculates a large supercell including all displacements - this supercell will be used to generate the phase gratings for each slice	
		//Here it is possible to overwrite the crystal structure with that following from a MD simulation
		Output_CIF(par->sc->cell, par->sc->name);				//Outputs the supercell as a cif file
		Init_GPU(par);											//Creates the element-wise phase cratings, writes data to GPU
		RunTurboSlice(par);										//start the multislice running	
	}	
	FreeGPUMemory(par);
	if (!par->thickness_scan) {
		average_pattern(par);
		ScaleImageForOutput(par);
		Output_TextImage(par, -1);	
		//Output_TextImageNoReflections(par);
		//Output_Reflection_List(par, -1);
		for (unsigned int i = 0; i < par->hkl.size(); i++) {par->hkl[i] = 0;}
	}

	cout << "DONE." << endl;
	MyExit();
	return 0;	
}


void RunTurboSlice(struct_parameters *par)	{

	cout << "Start CUDA processing" << endl;
	dim3 dimBlock( 16, 16 );
	dim3 dimGrid( par->nx / dimBlock.x, par->ny / dimBlock.y);

	for (int n = 0; n < par->totalslices; n++) {
		cout << "  Slice no " << (n + 1) << "/" << par->totalslices << " is running..." << endl;

		nullify<<<dimGrid, dimBlock>>>(par->nx, par->ny, par->GPU->d_trans);
		// calculate scattering amplitude - NOTE zero freg is in the bottom left corner and expandes into all other corners - not in the centre	this is required for FFT, high freq is in the centre
		for (unsigned int i = 0; i < par->sc->cell->atoms.size(); i++) {
			if (par->sc->cell->atoms[i].pos[2] >= ((float) n / (float) par->totalslices) && par->sc->cell->atoms[i].pos[2] < ((float) (n+1) / (float) par->totalslices)) {
				float y =  par->sc->cell->atoms[i].pos[1] * par->sc->cell->abc[1];
				float x = (par->sc->cell->atoms[i].pos[0] * par->sc->cell->abc[0]) + (cos(MyRad(par->sc->cell->angle[2])) * y);
				addatom <<<dimGrid, dimBlock>>> (x, y, par->GPU->d_kx, par->GPU->d_ky, par->nx, par->ny, par->CPU->link[par->sc->cell->atoms[i].atomic_no], par->GPU->d_trans, par->GPU->d_scat);				        
			} 
		}
//Only for debugging
//vector<Complex> trans; trans.resize(par->nx * par->ny);
//vector<Complex> scat; scat.resize(par->nx * par->ny * par->CPU->noofelements);
//if (cudaSuccess != cudaMemcpy(&scat[0], par->GPU->d_scat, scat.size() * sizeof(Complex), cudaMemcpyDeviceToHost)){printf( "Error Memcpy Wave Function back to Host!\n" );}	
//if (cudaSuccess != cudaMemcpy(&trans[0], par->GPU->d_trans, trans.size() * sizeof(Complex), cudaMemcpyDeviceToHost)){printf( "Error Memcpy Wave Function back to Host!\n" );}	
		//FFT
		cufftExecC2C(par->plan, (cufftComplex *) par->GPU->d_trans, (cufftComplex *) par->GPU->d_trans, CUFFT_INVERSE);
		//Required after FFT
		normalise<<<dimGrid, dimBlock>>>(par->nx, par->ny, par->GPU->d_trans);											
		//Change temporary array to the correct phase grating	
		convert<<<dimGrid, dimBlock>>>(par->nx, par->ny, par->GPU->d_trans, par->wavelen, par->mm0);				
		//Perform the phase grating convolution
		transmit<<<dimGrid, dimBlock>>>(par->GPU->d_wave, par->GPU->d_trans, par->nx, par->ny);	
		//transform to reciprocal space (is it?)
		cufftExecC2C(par->plan, (cufftComplex *) par->GPU->d_wave, (cufftComplex *) par->GPU->d_wave, CUFFT_FORWARD);				
		//propagate the wave-function
		propagate<<<dimGrid, dimBlock>>>(par->GPU->d_wave, par->GPU->d_prop, par->nx, par->ny);		
		//high k cutoff
		cutoff<<<dimGrid, dimBlock>>>(par->k2max, par->GPU->d_k2, par->nx, par->ny, par->GPU->d_wave);
		
		if( n + 1 == par->totalslices || par->thickness_scan) {		//If the electron wave is at the bottom surface of the crystal - or if(true) -> output every slice
			if (cudaSuccess != cudaMemcpy(&par->CPU->wave[0], par->GPU->d_wave, par->CPU->wave.size() * sizeof(Complex), cudaMemcpyDeviceToHost)){printf( "Error Memcpy Wave Function back to Host!\n" );}	//Transform back to real-space
			pattern_store(par);
		}	
		if 	(par->thickness_scan) {
			ScaleImageForOutput(par);
			Output_TextImage(par, (n+1));
			Output_Reflection_List(par, (n+1));
			for (unsigned int i = 0; i < par->hkl.size(); i++) {par->hkl[i] = 0;}				//During thickness scans we don't accumulate the values of several runs (max 1 run anyways for thickness scans) but write out a pattern for every slice
		}
		cufftExecC2C(par->plan, (cufftComplex *) par->GPU->d_wave, (cufftComplex *) par->GPU->d_wave, CUFFT_INVERSE);	
		normalise<<<dimGrid, dimBlock>>>(par->nx, par->ny, par->GPU->d_wave);
		}
}