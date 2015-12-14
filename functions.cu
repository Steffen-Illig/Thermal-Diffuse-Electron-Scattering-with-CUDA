/*					*** functions.cu ***
              	
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

/*----------------atomplot nullify()--------------------------*/
//reset the trans array for a new phase grating
__global__ void nullify( int nx, int ny, Complex *array) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	array[(ix * ny) + iy].re = 0.0F;
	array[(ix * ny) + iy].im = 0.0F;
} 		


/*----------------atomplot add()--------------------------*/
//add each fragment contribution to the reciprocal array
__global__ void addatom (float x, float y, float *d_kx, float *d_ky, int nx, int ny, int element, Complex *d_trans, Complex *d_scat ) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	float wr = d_scat[(element*nx*ny) + (iy * nx) + ix].re;
	float wi = d_scat[(element*nx*ny) + (iy * nx) + ix].im;
	
	float v = 2*PI*((d_kx[ix]*x) + (d_ky[iy]*y));
	
	d_trans[((iy * nx)+ix)].re += wr*cos(v) - wi*sin(v);
	d_trans[((iy * nx)+ix)].im += wr*sin(v) + wi*cos(v);
} 	


/*------------------------ propagate() ------------------------*/
//propagate the wavefunction thru one layer
__global__ void propagate(Complex *array, Complex *d_prop, int nx, int ny ) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	float wr = array[(iy * nx) + ix].re;
	float wi = array[(iy * nx) + ix].im;

	float propr = d_prop[(iy * nx) + ix].re;
	float propi = d_prop[(iy * nx) + ix].im;

	array[(iy * nx) + ix].re = wr*propr - wi*propi;
	array[(iy * nx) + ix].im = wr*propi + wi*propr;	
}

/*------------------------ transmit() ------------------------*/
//transmit the wavefunction thru one layer
__global__ void transmit( Complex *d_wave, Complex *d_trans, int nx, int ny ) {
	
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	float wr = d_wave[(iy * nx) + ix].re;
	float wi = d_wave[(iy * nx) + ix].im;
	float tr = d_trans[(iy * nx) + ix].re;
	float ti = d_trans[(iy * nx) + ix].im;

	d_wave[(iy * nx) + ix].re = wr*tr - wi*ti;
	d_wave[(iy * nx) + ix].im = wr*ti + wi*tr;
} 


/*--------------------atomplot cutoff() ---------------------*/
//Trim the transmission function to the spatial resolution of the microscope
__global__ void cutoff( float k2max, float *d_k2, int nx, int ny, Complex *array ) {	

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (d_k2[iy*nx + ix] > k2max) {
		array[(iy * nx) + ix].re = array[(iy * nx) + ix].im = 0.0F;
	} 
} 			


/*----------------atomplot normalise()--------------------------*/
//normalise after the Fourier transforms
__global__ void normalise(int nx, int ny, Complex *array) {
	
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
		
	array[(iy * nx)+ix].re /= (nx*ny);
	array[(iy * nx)+ix].im /= (nx*ny);
	
} 


/*----------------atomplot convert()--------------------------*/
//change temporary array to the correct phase grating	
__global__ void convert( int nx, int ny, Complex *array, float wavelen, float mm0 ) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
		
	float t = array[(iy * nx)+ix].re * wavelen * mm0;
	
	array[(iy * nx)+ix].re = cos(t);
	array[(iy * nx)+ix].im = sin(t);
	
} 


//------------------------------------------Non-GPU-Kernel Functions------------------------------------------

//Exit function
void MyExit() {
	system("PAUSE");
	exit(0);
}

//returns the square of a value
float MySq(float val) {
return (val*val);
}

//degree->rad
float MyRad(float deg) {
return (deg / (float) 180) * (float) PI;
}

//Rounds a value
int MyRound(float val) {
	if (val < 0) {
		return (int) ceil(val - 0.5);
		} else {
		return (int) floor(val + 0.5);
		}
}

string convertInt(int number) {
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

//removes all blanks from a string
string removeWhitespace(string str) {
    for (size_t i = 0; i < str.length(); i++) {
        if (str[i] == ' ') {
            str.erase(i, 1);
            i--;
        }
    }
    return str;
}

//String->atomic number
int ReturnAtomicNo (string element) {
	if (element == "H") {return 1;}
	if (element == "C") {return 6;}
	if (element == "N") {return 7;}
	if (element == "O") {return 8;}
	if (element == "F") {return 9;}
	if (element == "Si"){return 14;}
	if (element == "S") {return 16;}
	cout << "ELEMENT " << element << " NOT FOUND, ADD TO ReturnAtmoicNo-FUNCTION!\n";
	return -1;
	}

//calculates the volume of a triclinic unit cell
float CalculateVolume(struct_unitcellprop *cif) {
float a,b,c,al,be,ga;
a = cif->abc[0]; b = cif->abc[1]; c = cif->abc[2];
al = MyRad(cif->angle[0]); be = MyRad(cif->angle[1]); ga = MyRad(cif->angle[2]);
return a*b*c*sqrt(1 - MySq(cos(al)) - MySq(cos(be)) - MySq(cos(ga)) + 2*cos(al)*cos(be)*cos(ga));
}

//GetVectorMagnitude
float magnitude(vector<float> v) {
float mag = 0;
for (int i=0; i < 3; i++) {
    mag += v[i]*v[i];
    } 
mag = sqrt(mag);
return mag;
}

//Normalizes a vector
vector<float> normalize(vector<float> v) {
vector<float> result(3);
for (int i=0; i < 3; i++) {
    result[i] = v[i]/magnitude(v);
    } 
return result; 
}

//Add Vectors
vector<float> addvectors(vector<float> v, vector<float> w) {
	vector<float> z(3);
	z[0] = v[0] + w[0];
	z[1] = v[1] + w[1];
	z[2] = v[2] + w[2];
	return z;
}

//Subtract Vectors
vector<float> subtractvectors(vector<float> v, vector<float> w) {
	vector<float> z;
	z[0] = v[0] - w[0];
	z[1] = v[1] - w[1];
	z[2] = v[2] - w[2];
	return z;
}

//Dot Product
float dotproduct(vector<float> a, vector<float> b) {
	float prod = 0;
	for (int i = 0; i < 3; i++) {
		prod += a[i] * b[i];
	}
	return prod;
}

//Cross product
vector<float> crossproduct(vector<float> v, vector<float> w) {
	vector<float> z(3);
	z[0] = v[1] * w[2] - v[2] * w[1];
	z[1] = v[2] * w[0] - v[0] * w[2];
	z[2] = v[0] * w[1] - v[1] * w[0];
	return z;
}

//Returns a float vector describing the direction from atom1 to atom2 and length=1
vector<float> determine_direction_from_atoms(vector<float> atom1, vector<float> atom2) {
vector<float> result(3);
//Get direction
for (int i=0; i < 3; i++) {
    result[i] = atom2[i] - atom1[i];
    } 
//Normalize
result = normalize(result);
return result;
}

//Returns a float vector of length 1 that is perpendicular to the plane defined by atoms 1,2 & 3
vector<float> determine_direction_from_plane(vector<float> atom1, vector<float> atom2, vector<float> atom3) {
vector<float> result(3);
vector<float> direction1 = determine_direction_from_atoms(atom1, atom2);
vector<float> direction2 = determine_direction_from_atoms(atom1, atom3);
//Cross product
result = crossproduct(direction1, direction2);
//Normalize
result = normalize(result);
return result;
}

float random_gaussian(float mean, float sigma) {
//Box-Muller method
float a,b,val1,val2,max;
val1 = (float) rand();              //Conversion in Float
val2 = (float) rand();              //Conversion in Float
max = (float) RAND_MAX;             //Conversion in Float
a = (1 + val1)/(max+1);				//Avoiding 0 
b = (1 + val2)/(max+1);				//Avoiding 0  
float normal_distribution = sqrt(-2*log(a))*cos(2*b* (float) PI);
return (mean + normal_distribution*sigma);
}

void TriclinicToCartesian(struct_unitcellprop *cif) {
//http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm
float a, b, c, al, be, ga, vol, x, y, z;		//Makes the formulas easier
a = cif->abc[0]; b = cif->abc[1]; c = cif->abc[2];
al = MyRad(cif->angle[0]); be = MyRad(cif->angle[1]); ga = MyRad(cif->angle[2]);
vol = cif->volume;
for (unsigned int i = 0; i < cif->atoms.size(); i++) {
	x = cif->atoms[i].pos[0]; y = cif->atoms[i].pos[1]; z = cif->atoms[i].pos[2];		//necessary to temporarily store these
	cif->atoms[i].pos[0] = x*a + y*b*cos(ga) + z*c*cos(be);
	cif->atoms[i].pos[1] = 0   + y*b*sin(ga) + z*c*((cos(al) - cos(be)*cos(ga))/(sin(ga)));
	cif->atoms[i].pos[2] = 0   + 0           + z*vol/(a*b*sin(ga));
	}
}

void CartesianToTriclinic(struct_unitcellprop *cif) {
//http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm
float a, b, c, al, be, ga, vol, x, y, z;		//Makes the formulas easier
a = cif->abc[0]; b = cif->abc[1]; c = cif->abc[2];
al = MyRad(cif->angle[0]); be = MyRad(cif->angle[1]); ga = MyRad(cif->angle[2]);
vol = cif->volume;
for (unsigned int i = 0; i < cif->atoms.size(); i++) {
	x = cif->atoms[i].pos[0]; y = cif->atoms[i].pos[1]; z = cif->atoms[i].pos[2];		//necessary to temporarily store these
	cif->atoms[i].pos[0] = x/a + y*(-1)*cos(ga)/(a*sin(ga)) + z*(1/vol)*((b*cos(ga)*c*(cos(al)-cos(be)*cos(ga)))/sin(ga) - b*c*cos(be)*sin(ga));
	cif->atoms[i].pos[1] = 0   + y/(b*sin(ga))			  + z*(-1)*(a*c*(cos(al) - cos(be)*cos(ga)))/(vol*sin(ga));
	cif->atoms[i].pos[2] = 0   + 0						  + z*a*b*sin(ga)/vol;
	}
}

void AddPhonons(struct_unitcellprop *cif, struct_parameters *par) {
vector<float> dis(3); float ran;
for (unsigned int d = 0; d < par->sc->ph.size(); d++) {     //For all phonons
	ran = random_gaussian(0, par->sc->ph[d].sigma);			//Create displacement variable
	if (par->sc->ph[d].type == "trans") {//Translational Phonon
		for (int i = 0; i < 3; i++) { 
			dis[i] = ran * ( 
			par->sc->ph[d].d[0] * par->sc->sys[par->sc->ph[d].coordsystem].d[0][i] +
			par->sc->ph[d].d[1] * par->sc->sys[par->sc->ph[d].coordsystem].d[1][i] +
			par->sc->ph[d].d[2] * par->sc->sys[par->sc->ph[d].coordsystem].d[2][i]); 
		}
		for (unsigned int k = 0; k < par->cif->atoms.size(); k++) {
			if (cif->atoms[k].group == par->sc->ph[d].group) { cif->atoms[k].pos = addvectors(cif->atoms[k].pos, dis); }
		}
	} else if (par->sc->ph[d].type == "rot") {//Rotational Phonon
		//http://www.programming-techniques.com/2012/03/3d-rotation-algorithm-about-arbitrary.html for C++ code
		//Ran is the angle in rad - as the radius for core atoms is approx 1 Ang, ran is therefore approxmately the displacement in Ang, exaclty as for the translations
		float angle = ran;

		//u, v, w is the vector around which the rotation takes place
		float u, v, w;
		u = par->sc->ph[d].d[0] * par->sc->sys[par->sc->ph[d].coordsystem].d[0][0] +
			par->sc->ph[d].d[1] * par->sc->sys[par->sc->ph[d].coordsystem].d[1][0] +
			par->sc->ph[d].d[2] * par->sc->sys[par->sc->ph[d].coordsystem].d[2][0]; 
		v = par->sc->ph[d].d[0] * par->sc->sys[par->sc->ph[d].coordsystem].d[0][1] +
			par->sc->ph[d].d[1] * par->sc->sys[par->sc->ph[d].coordsystem].d[1][1] +
			par->sc->ph[d].d[2] * par->sc->sys[par->sc->ph[d].coordsystem].d[2][1]; 
		w = par->sc->ph[d].d[0] * par->sc->sys[par->sc->ph[d].coordsystem].d[0][2] +
			par->sc->ph[d].d[1] * par->sc->sys[par->sc->ph[d].coordsystem].d[1][2] +
			par->sc->ph[d].d[2] * par->sc->sys[par->sc->ph[d].coordsystem].d[2][2]; 

		//Calculate the rotation matrix
		float rotationMatrix[4][4];
		float L = (u*u + v * v + w * w);
		float u2 = u * u;
		float v2 = v * v;
		float w2 = w * w; 
 
		rotationMatrix[0][0] = (u2 + (v2 + w2) * cos(angle)) / L;
		rotationMatrix[0][1] = (u * v * (1 - cos(angle)) - w * sqrt(L) * sin(angle)) / L;
		rotationMatrix[0][2] = (u * w * (1 - cos(angle)) + v * sqrt(L) * sin(angle)) / L;
		rotationMatrix[0][3] = 0.0; 
 
		rotationMatrix[1][0] = (u * v * (1 - cos(angle)) + w * sqrt(L) * sin(angle)) / L;
		rotationMatrix[1][1] = (v2 + (u2 + w2) * cos(angle)) / L;
		rotationMatrix[1][2] = (v * w * (1 - cos(angle)) - u * sqrt(L) * sin(angle)) / L;
		rotationMatrix[1][3] = 0.0; 
 
		rotationMatrix[2][0] = (u * w * (1 - cos(angle)) - v * sqrt(L) * sin(angle)) / L;
		rotationMatrix[2][1] = (v * w * (1 - cos(angle)) + u * sqrt(L) * sin(angle)) / L;
		rotationMatrix[2][2] = (w2 + (u2 + v2) * cos(angle)) / L;
		rotationMatrix[2][3] = 0.0; 
 
		rotationMatrix[3][0] = 0.0;
		rotationMatrix[3][1] = 0.0;
		rotationMatrix[3][2] = 0.0;
		rotationMatrix[3][3] = 1.0;		

		//Here I am just following their conventions really to avoid stupid mistakes
		float inputMatrix[4][1] = {0.0, 0.0, 0.0, 0.0};
		float outputMatrix[4][1] = {0.0, 0.0, 0.0, 0.0};

		//I added this, the code requires the rotational axis to go through the centre, so we shift all the atoms by that distance, do the rotation and shift them back - this needs to be adjusted for a general axis as right now the rotational axis runs through the first defining point used to set up that particualr coordinate system
		float shift_x, shift_y, shift_z;
		shift_x = cif->atoms[par->sc->sys[par->sc->ph[d].coordsystem].definingatoms[0]].pos[0];
		shift_y = cif->atoms[par->sc->sys[par->sc->ph[d].coordsystem].definingatoms[0]].pos[1];
		shift_z = cif->atoms[par->sc->sys[par->sc->ph[d].coordsystem].definingatoms[0]].pos[2];

		for (unsigned int k = 0; k < par->cif->atoms.size(); k++) {
			if (cif->atoms[k].group == par->sc->ph[d].group) {
				inputMatrix[0][0] = cif->atoms[k].pos[0] - shift_x;
				inputMatrix[1][0] = cif->atoms[k].pos[1] - shift_y;
				inputMatrix[2][0] = cif->atoms[k].pos[2] - shift_z;
				inputMatrix[3][0] = 1.0;

				for(int i = 0; i < 4; i++ ){
					for(int j = 0; j < 1; j++){
						outputMatrix[i][j] = 0;
						for(int k = 0; k < 4; k++){
							outputMatrix[i][j] += rotationMatrix[i][k] * inputMatrix[k][j];
						}
					}
				}

				cif->atoms[k].pos[0] = outputMatrix[0][0] + shift_x;
				cif->atoms[k].pos[1] = outputMatrix[1][0] + shift_y;
				cif->atoms[k].pos[2] = outputMatrix[2][0] + shift_z;
			}
		}

	}
}
}

void AddRandomNoise(struct_unitcellprop *cif, struct_parameters *par) {
vector<float> dis(3);  //Displacements in x,y,z
for (unsigned int k = 0; k < par->cif->atoms.size(); k++) {//For all atoms int he unit cell...
	for (int d = 0; d < 3; d++) {     
		dis[d] = random_gaussian(0, par->sc->sigma_random_noise);//...create three random dispalcements...		 
		}
		cif->atoms[k].pos = addvectors(cif->atoms[k].pos, dis);//...and add them to the atmoic position
	}
}

void CompressShiftRename(struct_unitcellprop *cif, unsigned int u, unsigned int v, unsigned int w, unsigned int dim[]) {
	unsigned int ind[3] = {u, v, w};
	for (unsigned int k = 0; k < cif->atoms.size(); k++) {
		for (int i = 0; i < 3; i++) { 
			cif->atoms[k].pos[i] /= (float) dim[i];							//Compression
			cif->atoms[k].pos[i] += (float) ind[i] / (float) dim[i];		//Shift
			cif->atoms[k].label += "_" + convertInt(ind[i]);				//Rename
			}
		}
	}


vector<float> Correct(vector<float> pos) {//This ensures that atoms are correctly identified in their slice of the supercell
	//if (pos[0] <  0){pos[0] += 1;}	
	//if (pos[0] >= 1){pos[0] -= 1;}
	//if (pos[1] <  0){pos[1] += 1;}	
	//if (pos[1] >= 1){pos[1] -= 1;}
	if (pos[2] <  0){pos[2] += 1;}	
	if (pos[2] >= 1){pos[2] -= 1;}
	return pos;
}

//Creates the supercell including the displacements
void CalculateSuperCell(struct_parameters *par) {
	cout << "Creating supercell including all displacements. \n";
	par->sc->name = par->output + "_supercell-" + convertInt(par->sc->scsize[0]) + "x" + convertInt(par->sc->scsize[1]) + "x" + convertInt(par->sc->scsize[2]) + ".cif";	
	*par->sc->cell = *par->cif;		//Setting all angles which (identical as is cif file)
	par->sc->cell->abc[0] = par->cif->abc[0] * par->sc->scsize[0];
	par->sc->cell->abc[1] = par->cif->abc[1] * par->sc->scsize[1];
	par->sc->cell->abc[2] = par->cif->abc[2] * par->sc->scsize[2];
	par->sc->cell->volume = CalculateVolume(par->sc->cell);
	par->sc->cell->atoms.resize(par->sc->scsize[0]*par->sc->scsize[1]*par->sc->scsize[2]*par->cif->atoms.size());	
	struct_unitcellprop *tempcif = new struct_unitcellprop;
	for (unsigned int u = 0; u < par->sc->scsize[0]; u++) {
		for (unsigned int v = 0; v < par->sc->scsize[1]; v++) {
			for (unsigned int w = 0; w < par->sc->scsize[2]; w++) {	
				*tempcif = *par->cif;
				AddPhonons(tempcif, par);
				AddRandomNoise(tempcif, par);
				CartesianToTriclinic(tempcif);
				CompressShiftRename(tempcif, u, v, w, par->sc->scsize);
				unsigned int index = par->cif->atoms.size() * (w + (par->sc->scsize[2] * (v + (par->sc->scsize[1] * u))));
				for (unsigned int i = 0; i < par->cif->atoms.size(); i++) {
					par->sc->cell->atoms[index + i] = tempcif->atoms[i];
					par->sc->cell->atoms[index + i].pos = Correct(par->sc->cell->atoms[index + i].pos);//This ensures that atoms are correctly identified in their slice of the supercell
					}
				}
			}
		}
	}

void OverwriteSuperCell(struct_parameters *par, string source) {
	ifstream file;
	cout << "Overwriting Super Cell with MD-generated cell read from: " << source << endl;
	file.open(source.c_str(), ios::in);
	string read; 
	getline(file, read);	 
	//Jump to right position (where relevant data starts)
	while (! file.eof() && read.substr(0,19) != " _atom_site_fract_z") {
		getline(file, read);
	}
	int counter = 0;
	while (! file.eof()) {
	  getline(file, read);
	  if (!read.empty()) {			 
		 par->sc->cell->atoms[counter].label = removeWhitespace(read.substr(0, read.find("\t"))); read = read.substr(read.find("\t") + 1, read.length()-(read.find("\t") + 1));
		 par->sc->cell->atoms[counter].type = removeWhitespace(read.substr(0, read.find("\t"))); read = read.substr(read.find("\t") + 1, read.length()-(read.find("\t") + 1));
		 par->sc->cell->atoms[counter].atomic_no = ReturnAtomicNo(par->sc->cell->atoms[counter].type);
		 par->sc->cell->atoms[counter].pos[0] = stof(read.substr(0, read.find("\t"))); read = read.substr(read.find("\t") + 1, read.length()-(read.find("\t") + 1));
		 par->sc->cell->atoms[counter].pos[1] = stof(read.substr(0, read.find("\t"))); read = read.substr(read.find("\t") + 1, read.length()-(read.find("\t") + 1));
		 par->sc->cell->atoms[counter].pos[2] = stof(read.substr(0, read.find("\t"))); read = read.substr(read.find("\t") + 1, read.length());
		 par->sc->cell->atoms[counter].pos = Correct(par->sc->cell->atoms[counter].pos);	//This ensures that atoms are correctly identified in their slice of the supercell
		 counter++;
         }
      }
	if (counter != par->sc->cell->atoms.size()){cout << "Error: MD-cif-file contains wrong number of atoms.\n";}
	file.close();
	file.clear();
}

void average_pattern(struct_parameters *par) {	
	cout << "Average the hkl values of all simulations.\n";
	for (unsigned int i = 0; i < par->hkl.size(); i++) {
		par->hkl[i] /= par->averaging;
	}
}

void pattern_store(struct_parameters *par) {
	cout << "Storing hkl values internally.\n";
	float re, im;
	for (int j=0; j < par->ny; j++) {
		for (int i=0; i < par->nx; i++ ) {
			if (j < par->ny/2 ) {
				if (i < par->nx/2) {						
					re = par->CPU->wave[i+(par->nx/2)+(par->nx*(j+(par->ny/2)))].re;
					im = par->CPU->wave[i+(par->nx/2)+(par->nx*(j+(par->ny/2)))].im;
				} else {
					re = par->CPU->wave[i-(par->nx/2)+(par->nx*(j+(par->ny/2)))].re;
					im = par->CPU->wave[i-(par->nx/2)+(par->nx*(j+(par->ny/2)))].im;
				}
			} else {
				if (i < par->nx/2) {						
					re = par->CPU->wave[i+(par->nx/2)+(par->nx*(j-(par->ny/2)))].re;
					im = par->CPU->wave[i+(par->nx/2)+(par->nx*(j-(par->ny/2)))].im;
				} else {
					re = par->CPU->wave[i-(par->nx/2)+(par->nx*(j-(par->ny/2)))].re;
					im = par->CPU->wave[i-(par->nx/2)+(par->nx*(j-(par->ny/2)))].im;
				}
			}	
			par->hkl[j*par->ny + i] += sqrt(re*re + im*im);
		}
	}
}

void FreeGPUMemory (struct_parameters *par) {
	cudaFree(par->GPU->d_k2);
	cudaFree(par->GPU->d_kx);
	cudaFree(par->GPU->d_ky);
	cudaFree(par->GPU->d_prop);
	cudaFree(par->GPU->d_trans);
	cudaFree(par->GPU->d_wave);
	cudaFree(par->GPU->d_scat);
}

#define PI 3.14159265358979323846 											


