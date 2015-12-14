/*					*** init.cu ***
              	
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

#include "functions.h"
#include "structs.h"

using namespace std;

#define PI 3.14159265358979323846


//Create all parameter-internal structs
void Init_General (struct_parameters *par) {

	struct_supercell *sc = new struct_supercell; 
	par->sc = sc;
	struct_unitcellprop *cell = new struct_unitcellprop;
	par->sc->cell = cell;
	struct_CPU_data *CPU = new struct_CPU_data; 
	par->CPU = CPU;
	struct_GPU_data *GPU = new struct_GPU_data;
	par->GPU = GPU;
	}


//Read the structure file (modified .cif file that contains all atoms and structural parameter)
void Init_ReadStructureFile(struct_parameters *par) {

	struct_unitcellprop *cif = new struct_unitcellprop;
	par->cif = cif;

	string source = par->prefix + "_structure.cif";
	cout << "Reading " << source << endl;
	ifstream file(source);
	if( !file ) {
       cout << "Error reading " << source << ".\n";
       }	
	string read;
	unsigned int length;
	getline(file, read);																				//1. line is title and can be discarded
	getline(file, read); if (removeWhitespace(read) != ""){cout << "Cif file not in expected format (1). \n"; MyExit();}			//2. line is blank
	//Reading a,b,c,alpha,beta,gamma
	read = "_cell_length_a"; length = read.length(); getline(file, read);
	cif->abc[0] = stof(read.substr(length, read.length() - length));
	read = "_cell_length_b"; length = read.length(); getline(file, read);
	cif->abc[1] = stof(read.substr(length, read.length() - length));
	read = "_cell_length_c"; length = read.length(); getline(file, read);
	cif->abc[2] = stof(read.substr(length, read.length() - length));
	read = "_cell_length_alpha"; length = read.length(); getline(file, read);
	cif->angle[0] = stof(read.substr(length, read.length() - length));
	read = "_cell_length_beta"; length = read.length(); getline(file, read);
	cif->angle[1] = stof(read.substr(length, read.length() - length));
	read = "_cell_length_gamma"; length = read.length(); getline(file, read);
	cif->angle[2] = stof(read.substr(length, read.length() - length));

	//Calculating and reading the volume (comparison to detect parameter errors)
	read = "_cell_volume"; length = read.length(); getline(file, read);
	float volume = stof(read.substr(length, read.length() - length));
	cif->volume = abs(CalculateVolume(cif));
	if ((cif->volume - volume) > 1) {
		cout << "WARNING: Cif unit cell volume doesn't match the calculated one. \n";
		cout << "Cif-File Volume: " << volume << "       Calculated Volume: " << CalculateVolume(cif) << endl;
		cout << "This might be an indication that structural input parameters have errors. \n"; 
		}

	getline(file, read); if (removeWhitespace(read) != ""){cout << read << "Cif file not in expected format (2). \n"; MyExit();}	
	getline(file, read); if (removeWhitespace(read) != "loop_"){cout << "Cif file not in expected format (3). \n"; MyExit();}	
	getline(file, read); if (removeWhitespace(read) != "_atom_site_label"){cout << "Cif file not in expected format (4). \n"; MyExit();}	
	getline(file, read); if (removeWhitespace(read) != "_atom_site_type_symbol"){cout << "Cif file not in expected format (5). \n"; MyExit();}	
	getline(file, read); if (removeWhitespace(read) != "_atom_site_fract_x"){cout << "Cif file not in expected format (6). \n"; MyExit();}	
	getline(file, read); if (removeWhitespace(read) != "_atom_site_fract_y"){cout << "Cif file not in expected format (7). \n"; MyExit();}	
	getline(file, read); if (removeWhitespace(read) != "_atom_site_fract_z"){cout << "Cif file not in expected format (8). \n"; MyExit();}	

	struct_atom *temp = new struct_atom;
	while (! file.eof()) {
	  getline(file, read); 
	  if (!read.empty()) {
		 //cout << read << endl; 
		 temp->label = removeWhitespace(read.substr(0, read.find("\t"))); read = read.substr(read.find("\t") + 1, read.length()-(read.find("\t") + 1));
		 //cout << temp->label << "\n";
		 temp->type = removeWhitespace(read.substr(0, read.find("\t"))); read = read.substr(read.find("\t") + 1, read.length()-(read.find("\t") + 1));
		 //cout << temp->type << "\n";
		 temp->atomic_no = ReturnAtomicNo(temp->type);
		 //cout << temp->atomic_no << endl;
		 temp->pos[0] = stof(read.substr(0, read.find("\t"))); read = read.substr(read.find("\t") + 1, read.length()-(read.find("\t") + 1));
		 //cout << temp->pos[0] << endl;
		 temp->pos[1] = stof(read.substr(0, read.find("\t"))); read = read.substr(read.find("\t") + 1, read.length()-(read.find("\t") + 1));
		 //cout << temp->pos[1] << endl;
		 temp->pos[2] = stof(read.substr(0, read.find("\t"))); read = read.substr(read.find("\t") + 1, read.length());
		 //cout << temp->pos[2] << endl << endl;
         par->cif->atoms.push_back(*temp);
		 }
      }
}


void Init_ReadInputFile(struct_parameters *par) {

	const string source = par->prefix + "_input.txt";
	cout << "Reading " << source << endl;
	//Nested scope will release ifstream resource when no longer needed	
	ifstream file(source);
	string read;
	if( !file ) {
       std::cout << "Error reading " << source << ".\n";
       }

	//Suffix for ouput file:
	getline(file, read); cout << "  " << "Output file: ";
	getline(file, read); par->output = par->prefix + "_" + read; cout << par->output << endl; 	
	
	//Averaging output over X simulations - Runs the simulation several times and averages the final output
	getline(file, read); cout << "  " << read << " ";
	getline(file, read); par->averaging = stoi(read); cout << par->averaging << endl;

	//Is this a thickness scan?
	getline(file, read); cout << "  Thickness scan?: ";
	getline(file, read); read = removeWhitespace(read);
	if (read == "yes" || read == "Yes" || read == "YES" || read == "true" || read == "True" || read == "TRUE") {
		par->thickness_scan = true; cout << "Yes" << endl;
	} else {
		par->thickness_scan = false; cout << "No" << endl;
	}

	//Reading Acceleration Voltage	
	getline(file, read); cout << "  " << read << " ";
	getline(file, read); par->tension = stof(read); cout << par->tension << endl; 
	const float emass = (float) 510.99906; 		// electron rest mass in keV
	const float hc = (float) 12.3984244; 		// Planck's const x speed of light
	par->wavelen = (float) (hc/sqrt(par->tension * (2*emass + par->tension)));
	par->mm0 = (float) (1 + (par->tension/emass));
	
	//Size of Simulated Super Cell
	getline(file, read); cout << "  " << read << " ";
	getline(file, read); par->sc->scsize[0] = stoi(read); cout << par->sc->scsize[0] << " x ";
	getline(file, read); par->sc->scsize[1] = stoi(read); cout << par->sc->scsize[1] << endl;

	//Total number of layers to propagate through
	getline(file, read); cout << "  " << read << " " ;
	getline(file, read); par->sc->scsize[2] = stoi(read); cout << par->sc->scsize[2] << endl; 
	
	//Approximate thickness of each slice in Angstroem (will be adjusted later so that a multiple equals the crystal thickness)
	getline(file, read); cout << "  " << read << " " ;
	getline(file, read); par->slice_thickness = stof(read); cout << par->slice_thickness << endl; 
	
	//Sigma of the random noise affecting all atoms
	getline(file, read); cout << "  " << read << " " ;
	getline(file, read); par->sc->sigma_random_noise = stof(read); cout << par->sc->sigma_random_noise << endl; 
	
	//Debye-Waller-Factor (non-standard defintion)
	getline(file, read); cout << "  " << read << " " ;
	getline(file, read); par->DWF = stof(read); cout << par->DWF << endl; 

	//Size of Arrays, output structure and maximum h and k indeces:
	getline(file, read); cout << "  " << read << " ";
	getline(file, read); par->nx = stoi(read); cout << par->nx << " x ";
	getline(file, read); par->ny = stoi(read); cout << par->ny << endl; 

	//Reading in atoms to calculate the coordinate systems used to describe the displacements
	getline(file, read); cout << "  " << read << "\n";
	getline(file, read); 
	struct_coordinatesystem *tempcoord = new struct_coordinatesystem;
	while (read.substr(0, 18) == "Coordinate System:") { 
		cout << "  Coordinate System: ";
		read = read.substr(18, read.length() - 18);
		tempcoord->definingatoms[0] = stoi(read.substr(0, read.find(","))); cout << tempcoord->definingatoms[0] << ", "; read = read.substr(read.find(",") + 1, read.length()-(read.find(",") + 1));
		tempcoord->definingatoms[1] = stoi(read.substr(0, read.find(","))); cout << tempcoord->definingatoms[1] << ", "; read = read.substr(read.find(",") + 1, read.length()-(read.find(",") + 1));
		tempcoord->definingatoms[2] = stoi(read.substr(0, read.find(","))); cout << tempcoord->definingatoms[2] << "\n"; 
		par->sc->sys.push_back(*tempcoord);
		getline(file, read);
	} 
	
	//Reading in the phonon information
	cout << "  " << read << "\n";
	getline(file, read); 
	struct_phonons *tempph = new struct_phonons;
	while (read.substr(0, 7) == "Phonon:") { 
		cout << "  Phonons: ";
		read = read.substr(7, read.length() - 7);
		tempph->type		= removeWhitespace(read.substr(0, read.find(","))); cout << tempph->type << ", ";			read = read.substr(read.find(",") + 1, read.length()-(read.find(",") + 1));
		tempph->sigma		= stof(read.substr(0, read.find(",")));				cout << tempph->sigma << ", ";			read = read.substr(read.find(",") + 1, read.length()-(read.find(",") + 1));
		tempph->group		= stoi(read.substr(0, read.find(",")));				cout << tempph->group << ", ";			read = read.substr(read.find(",") + 1, read.length()-(read.find(",") + 1));
		tempph->coordsystem = stoi(read.substr(0, read.find(",")));				cout << tempph->coordsystem << ", ";	read = read.substr(read.find(",") + 1, read.length()-(read.find(",") + 1));
		tempph->d[0]		= stof(read.substr(0, read.find(",")));				cout << tempph->d[0] << ", ";			read = read.substr(read.find(",") + 1, read.length()-(read.find(",") + 1));
		tempph->d[1]		= stof(read.substr(0, read.find(",")));				cout << tempph->d[1] << ", ";			read = read.substr(read.find(",") + 1, read.length()-(read.find(",") + 1));
		tempph->d[2]		= stof(read.substr(0, read.find(",")));				cout << tempph->d[2] << "\n"; 
		par->sc->ph.push_back(*tempph);
		getline(file, read);
	} 	

	//Groups of interest (defining different subgroups across the unit cell that can be assigned to different phonons)
	cout << "  " << read << "\n";
	for (unsigned int i = 0; i < par->cif->atoms.size(); i++) {
		getline(file, read); 
		par->cif->atoms[i].group = stoi(read); 
		cout << par->cif->atoms[i].group << ",";
		}
	cout << endl;
}


//Processes Input data
void Init_ProcessInputData(struct_parameters *par) {
	
	cout << "Processing input data. \n";
	
	//Check User Input
	if (par->thickness_scan && par->averaging > 1) {
		cout << "No thickness scan possible if averaging > 1 --> change input variables!\n";
		MyExit();
	}
	
	//Initializes the hkl array that will store the output values
	par->hkl.resize(par->nx * par->ny);
	for (unsigned int i = 0; i < par->hkl.size(); i++) {
		par->hkl[i] = 0;
	}

	//Transform Unit Cell into cartesian coordinates
	TriclinicToCartesian(par->cif);
	
	//Normalize the direction vector for each phonon
	for (unsigned int i = 0; i < par->sc->ph.size(); i++) {
		par->sc->ph[i].d = normalize(par->sc->ph[i].d); 
	}
	
	//Derive Coordinate systems from the specified atoms
	for (unsigned int i = 0; i < par->sc->sys.size(); i++) {
		par->sc->sys[i].d.resize(3);
		par->sc->sys[i].d[0].resize(3); 
		par->sc->sys[i].d[0] = determine_direction_from_atoms(par->cif->atoms[par->sc->sys[i].definingatoms[0]].pos, par->cif->atoms[par->sc->sys[i].definingatoms[1]].pos);
		par->sc->sys[i].d[2].resize(3); 
		par->sc->sys[i].d[2] = determine_direction_from_plane(par->cif->atoms[par->sc->sys[i].definingatoms[0]].pos, par->cif->atoms[par->sc->sys[i].definingatoms[1]].pos, par->cif->atoms[par->sc->sys[i].definingatoms[2]].pos);
		par->sc->sys[i].d[1].resize(3); 
		par->sc->sys[i].d[1] = normalize(crossproduct(par->sc->sys[i].d[0], par->sc->sys[i].d[2]));
		}

	//Adjusts slice thicknessso that super cell thickness is a multiple
	//If we view the crystal along [001] the electron beam is along the (triclinic) c axis - this will have to be changed when arbitrary axis are used
	cout << "  Thickness of slices has been adjusted: " << par->slice_thickness << " Ang --> ";		
	par->totalslices = MyRound(par->sc->scsize[2] * par->cif->abc[2] / par->slice_thickness);
	par->slice_thickness = par->sc->scsize[2] * par->cif->abc[2] / (float) par->totalslices;	//This needs to be adjusted when the viewing direction is not [001] anymore
	cout << par->slice_thickness << " Ang \n";
	cout << "  Resulting in a total of " <<  par->totalslices << " slices of the same thickness.\n";
}

//Creates and writes the scat arrays & link function to GPU
void Init_GPU (struct_parameters *par) {

	cout << "Init GPU arrays... \n";
	//Create the arrays of reciprocal space vectors 
	vector<float> kx, ky, k2;
	kx.resize(par->nx); 
	ky.resize(par->ny);
	k2.resize(par->nx * par->ny);
	for(int i=0; i < par->nx; i++) {		
		if ( i > (par->nx / 2) ) {
			kx[i] = ((float) (i - par->nx)) / (float) par->sc->cell->abc[0];
		} else {
			kx[i] = ((float) i) / (float) par->sc->cell->abc[0];
		}		
		for(int j=0; j < par->ny; j++) {
			if ( j > (par->ny / 2) ) {
				ky[j] = ((float) (j - par->ny)) / (float) par->sc->cell->abc[1];
			} else {
				ky[j] = ((float) j) / (float) par->sc->cell->abc[1];
			}
			k2[(j*par->ny)+i] = MySq(kx[i]) + MySq(ky[j]);
		}
	}	
	//Write reciprocal space vectors to GPU
	if (cudaMalloc(&par->GPU->d_kx, kx.size() * sizeof(float)) != cudaSuccess) {cout << "CudaMalloc failed for d_kx./n" << endl; MyExit();}
	if (cudaMalloc(&par->GPU->d_ky, ky.size() * sizeof(float)) != cudaSuccess) {cout << "CudaMalloc failed for d_ky./n" << endl; MyExit();}	
	if (cudaMalloc(&par->GPU->d_k2, k2.size() * sizeof(float)) != cudaSuccess) {cout << "CudaMalloc failed for d_k2./n" << endl; MyExit();}
	cudaMemcpy(par->GPU->d_kx, &kx[0], kx.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(par->GPU->d_ky, &ky[0], ky.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(par->GPU->d_k2, &k2[0], k2.size() * sizeof(float), cudaMemcpyHostToDevice);

	//k2max is needed for cut off of highest frequencies
	float x_cut = par->nx / (2 * par->cif->abc[0] * par->sc->scsize[0]); 
	float y_cut = par->ny / (2 * par->cif->abc[1] * par->sc->scsize[1]);
	if( y_cut < x_cut ){
		par->k2max = MySq(y_cut * (float) 0.75);
	} else {
		par->k2max = MySq(x_cut * (float) 0.75);
	}

	//Count Elements present in structure and create the link function (stored in CPU)
	par->CPU->link.resize(100);
	for (unsigned int i = 0; i < par->CPU->link.size(); i++) {par->CPU->link[i] = -1;}
	par->CPU->noofelements = 0;
	for (unsigned int i = 0; i < par->cif->atoms.size(); i++) {
		if (par->CPU->link[par->cif->atoms[i].atomic_no] == -1) {
			par->CPU->link[par->cif->atoms[i].atomic_no] = par->CPU->noofelements;
			par->CPU->noofelements++;
			}
		}
	

	//Read atomic data table fparams.dat - I took this function basically without any changes from Alex' code
	vector<vector<float> > fparams;			//If we have 3 elements, e.g. C, Si and S the fparams dimensions will be fparams[3][na=14]
	const int nl = 3, ng = 3;				//used to read the fe-table	= number of Lorenzians and Gaussians
	const int na = 2*( nl + ng ) + 2;		//used to read the fe-table	= number of params to fit
	char cline[132], *cstatus;
	float z, chisqrd;
	FILE *fp;
	if( (fp = fopen( "fparams.dat", "r") ) != NULL) {} 							// open input file to fp
	else {
		cout << "ERROR - Can't open file fparams.dat \n"; 						// cannot find file
		MyExit();
	}
	fparams.resize(par->CPU->noofelements);												//For all elements			   
	for(int l = 0; l < par->CPU->noofelements; l++){ fparams[l].resize(na); }				// fparams includes z,chisq,L1,L1',L2,L2' ... G3,G3'
	for(int zi = 1; zi < 100; zi++) { 											// cycle through the whole table from element 1 to 99																				
		do {																	// find Z delimiter
			cstatus = fgets( cline, 132, fp );
			if( cstatus == NULL ) break;
		}
		while ( strncmp( cline, "Z=", 2 ) != 0 );
		if( cstatus == NULL ) break;
		sscanf( cline, "Z=%f,  chisq=%f\n", &z, &chisqrd);
		if(par->CPU->link[zi] != -1){ 														// only when zi = element in table
			fparams[par->CPU->link[zi]][0] = (float) z; 										// 0th element of each "row" = atomic number
			fparams[par->CPU->link[zi]][1] = chisqrd; 										// 1st element of each "row" = chi sqrd
			for(int j=0; j<12; j+=4 ) {
				fgets( cline, 132, fp );
				for(int i=0; i<4; i++) {																	
					sscanf(&cline[i*17],"%f", &fparams[par->CPU->link[zi]][i+j+2] );			// assign Ls and Gs to indexes 2 to 13
				}
			}
		}
	} 																			
	fclose( fp );


	//With fparams it is now possible to calculate the atomic form factors
	vector<Complex> scat;
	scat.resize(par->CPU->noofelements * par->nx * par->ny);
	for(int zi = 1; zi < 100; zi++) { 
		if(par->CPU->link[zi] != -1){ 
			int index = par->CPU->link[zi];
			for (int ix = 0; ix < par->nx; ix++) {
				for (int iy = 0; iy < par->ny; iy++) {
					float fe = 0;									// Atom form factor																					
					for(int i = 0; i < 2*nl; i+=2){					// Lorenztians
						fe += fparams[index][i+2] / (k2[par->nx*iy + ix] + fparams[index][i+3]);		
					}
					for(int i = 2*nl; i < 2*(nl + ng); i+=2){		// Gaussians
						fe += fparams[index][i+2] * exp( -k2[par->nx*iy + ix] * fparams[index][i+3]);
						}
					//float w = 2*PI*((kx[ix] * d_x[q] * d_ax) + (ky[iy] * d_y[q] * d_by));		//exponent of the structure factor, kx/ky is g, d_x[q] is r, d_ax is scaling parameter --> not needed as d_q is (0,0) and thus w=0	
					//I assumed when I rewrote Alex' code that these arrays that we write here (ctempdynamic in Alex code) were at this point ONLY initialized to zero via big array
					//The reason I am writing this is because Alex assigned the values via += which I turned into a = in the code below. Just keep this in mind when debugging.
					scat[(index * par->nx * par->ny) + (par->nx * iy) + ix].re = fe * par->nx * par->ny / (par->sc->cell->abc[0] * par->sc->cell->abc[1]);	//cos(w) * d_scale * fe; --> w=0
					scat[(index * par->nx * par->ny) + (par->nx * iy) + ix].re *= (float) exp(- par->DWF * k2[par->nx*iy + ix]);
					scat[(index * par->nx * par->ny) + (par->nx * iy) + ix].im = 0; //sin(w) * d_scale * fe; --> w=0
				}
			}
		}
	}
	//Write scat to GPU
	if (cudaMalloc(&par->GPU->d_scat, scat.size() * sizeof(Complex)) != cudaSuccess) {cout << "CudaMalloc failed for d_scat./n" << endl; MyExit();}
	cudaMemcpy(par->GPU->d_scat, &scat[0], scat.size() * sizeof(Complex), cudaMemcpyHostToDevice);

	//Create the cufft plan for the whole program a la fftw										
	cufftPlan2d(&par->plan, par->nx, par->ny, CUFFT_C2C); 

	//Initialise the transmission function to COMPLEX(0,0) and write to GPU
	vector<Complex> trans;
	trans.resize(par->nx * par->ny);
	if (cudaSuccess != cudaMalloc((void**)&par->GPU->d_trans, trans.size() * sizeof(Complex))) {printf( "Error Malloc Transmission Function!\n" );}
	for(int ix=0; ix < par->nx; ix++) {															
		for(int iy=0; iy < par->ny; iy++) {
			trans[(iy*par->nx)+ix].re = 0;
			trans[(iy*par->nx)+ix].im = 0;
		}
	}
	if (cudaSuccess != cudaMemcpy(par->GPU->d_trans, &trans[0], trans.size() * sizeof(Complex), cudaMemcpyHostToDevice) ){printf( "Error Memcpy Transmission Function!\n" );}
	

	//Initialise the electron wave function to COMPLEX unity (1,0) and write to GPU
	par->CPU->wave.resize(par->nx * par->ny);
	if ( cudaSuccess != cudaMalloc((void**)&par->GPU->d_wave, par->CPU->wave.size() * sizeof(Complex)) ){printf( "Error Malloc Electron Wave Function!\n" );}
	for(int ix=0; ix < par->nx; ix++) {														
		for(int iy=0; iy < par->ny; iy++) {
			par->CPU->wave[(iy * par->nx) + ix].re = 1; 
			par->CPU->wave[(iy * par->nx) + ix].im = 0; 
 			}
 	}
    if ( cudaSuccess != cudaMemcpy(par->GPU->d_wave, &par->CPU->wave[0], par->CPU->wave.size() * sizeof(Complex), cudaMemcpyHostToDevice) ){printf( "Error Memcpy Electron Wave Function!\n" );}


	//Calculate the propagation array and write to GPU
	vector<Complex> prop;
	prop.resize(par->nx * par->ny);

	if (cudaSuccess != cudaMalloc((void**)&par->GPU->d_prop, prop.size() * sizeof(Complex))) {printf( "Error Malloc Electron Propagation Function!\n" );}
	for (int ix = 0; ix < par->nx; ix++) {					
		float x = (float) PI * par->slice_thickness *par->wavelen * MySq(kx[ix]);
		for(int iy = 0; iy < par->ny; iy++) {
			float y = (float) PI * par->slice_thickness * par->wavelen * MySq(ky[iy]);
			prop[iy*par->nx + ix].re = (cos(x) *  cos(y)) - (-sin(x) * -sin(y));
			prop[iy*par->nx + ix].im = (cos(x) * -sin(y)) + (-sin(x) *  cos(y));
		}
	}	
    if ( cudaSuccess != cudaMemcpy(par->GPU->d_prop, &prop[0], prop.size() * sizeof(Complex), cudaMemcpyHostToDevice) ){printf( "Error Malloc Electron Propagation Function!\n" );}	
}