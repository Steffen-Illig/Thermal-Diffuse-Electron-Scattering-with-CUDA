/*					*** output.cu ***
              	
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

#ifndef _output_H_
#define _output_H_


void Output_CIF (struct_unitcellprop *cif, string filename) {
	cout << "Writing super cell as:\n  " << filename << endl;
	string InternalName = "data_" + filename;
	FILE * pFile;
	fopen_s(&pFile, filename.c_str(), "w");
	//Writing Header Information
	fprintf(pFile, "%s\n\n _cell_length_a \t%10.6f\n _cell_length_b \t%10.6f\n _cell_length_c \t%10.6f\n", InternalName.c_str(), cif->abc[0], cif->abc[1], cif->abc[2]);
	fprintf(pFile, " _cell_angle_alpha \t%10.6f\n _cell_angle_beta \t%10.6f\n _cell_angle_gamma \t%10.6f\n", cif->angle[0], cif->angle[1], cif->angle[2]);
	fprintf(pFile, " _cell_volume \t%10.6f\n\n", cif->volume);
	fprintf(pFile, "loop_ \n _atom_site_label \n _atom_site_type_symbol \n _atom_site_fract_x \n _atom_site_fract_y \n _atom_site_fract_z \n");
	//Writing Atomic Coordinates
	for (unsigned int i = 0; i < cif->atoms.size(); i++) {
		fprintf(pFile, "%s\t%s\t%10.6f\t%10.6f\t%10.6f\n", cif->atoms[i].label.c_str(), cif->atoms[i].type.c_str(), 
		cif->atoms[i].pos[0], cif->atoms[i].pos[1], cif->atoms[i].pos[2]);
	}
	fclose(pFile);
}


//Scales Image as 32Bit unsigned int
void ScaleImageForOutput(struct_parameters *par) {

	float max = 0;
	for (int y = 0; y < par->ny; y++) {
		for (int x = 0; x < par->nx; x++) { 
			if (par->hkl[y*par->ny + x] > max){max = par->hkl[y*par->ny + x];}
		}
	}
	float scaling = (float) 2000000000 / max;
	for (int y = 0; y < par->ny; y++) {
		for (int x = 0; x < par->nx; x++) { 
			par->hkl[y*par->ny + x] = (float) MyRound(par->hkl[y*par->ny + x] * scaling);
		}
	}
}

//Output as 32bit TextImage which even allows float values
void Output_TextImage (struct_parameters *par, int slice) {
	
	string output;
	if (slice == -1){
		output = par->output + "_TextIm.txt";
	} else {
		output = par->output + "_Slice-" + convertInt(slice) + "of" + convertInt(par->totalslices) + "-Slices-with-" + to_string(par->slice_thickness) + "AngEach_TextIm.txt";		
	}
	//output = convertInt((int) (par->var1 * 1000)) + "-" + convertInt((int) (par->var2 * 1000)) +  "-" + convertInt((int) (par->var3 * 1000)) + ".txt";
	cout << "Writing data as Text Image in: " << output << endl;
	FILE * pFile;
	fopen_s(&pFile, output.c_str(), "w");

	for (int y = 0; y < par->ny; y++) {
		for (int x = 0; x < par->nx; x++) { 
			fprintf(pFile, "%f\t", par->hkl[y*par->ny + x]);
		}
		fprintf(pFile, "\n");
	}	
	fclose(pFile);
}


//pattern_print but all reflections are set to the average of their 4 surrounding pixels
void Output_TextImageNoReflections(struct_parameters *par) {	

	string output = par->output + "NoReflTextIm.txt";
	cout << "Writing hkl output without reflections as:\n  " << output << endl;

	FILE * pFile;
	fopen_s(&pFile, output.c_str(), "w");

	int centre_x = par->nx/2;
	int centre_y = par->ny/2;
	int offset_x = MyRound(cos(MyRad(par->sc->cell->angle[2])) * par->sc->cell->abc[1] / par->cif->abc[0]); // E.g. for TES this is -3!
	for (int y = 0; y < par->ny; y++) {
		for (int x = 0; x < par->nx; x++) { 
			bool reflection = false;
			if (abs(y - centre_y) % (int) par->sc->scsize[1] == 0) {//only in this row (y value) are reflections
				int offset_x_this_y = offset_x * ((y - centre_y) / (int) par->sc->scsize[1]);
				if (abs(x - (centre_x - offset_x_this_y)) % (int) par->sc->scsize[0] == 0) {//for this row, there are only reflection in these columns (x)
					reflection = true;
				}
			}
			if (reflection && x > 0 && y > 0 && x < (par->nx - 1) && y < (par->ny - 1)) {
				float value = (par->hkl[y*par->ny + x+1] + par->hkl[y*par->ny + x-1] + par->hkl[(y+1)*par->ny + x] + par->hkl[(y-1)*par->ny + x]) / (float) 4;
				fprintf(pFile, "%f\t", value);
			} else {
				fprintf(pFile, "%f\t", par->hkl[y*par->ny + x]);
			}
		}
		fprintf(pFile, "\n");
	}
	fclose(pFile);
}


//Outputs all reflections (value taken above their "noise" level) as a h-k list
void Output_Reflection_List(struct_parameters *par, int slice) {	

	string output;
	if (slice == -1){
		output = par->output + "_ReflList.txt";
	} else {
		output = output = par->output + "_Slice-" + convertInt(slice) + "of" + convertInt(par->totalslices) + "-Slices-with-" + to_string(par->slice_thickness) + "_ReflList.txt";		
	}
	cout << "Writing reflection list as:\n  " << output << endl;
	
	FILE *outfile;
	outfile = fopen(output.data(), "w+");
	fprintf(outfile, "%s\t%s\t%s\n", "h","k", "Intensity");
	//ImageJ takes y coordinates from the top whereas in the bmp out put the start at the bottom, therefore: y = (par->ny-1) - ImageJ_y
	//E.g. the centre (0,0) lies at x = par->nx/2, y = par->ny/2 (e.g. (256,256) so in ImageJ at (256,255))
	//Goal is a tab seperated output list stating the cutoff frequnecy in the head line and for every subsequent line the h&k indices and the corresponding intensity
	int centre_x = par->nx/2;
	int centre_y = par->ny/2;
	int offset_x = MyRound(cos(MyRad(par->sc->cell->angle[2])) * par->sc->cell->abc[1] / par->cif->abc[0]); // E.g. for TES this is -3!
	for (int y = 1; y < par->ny-1; y++) {
		for (int x = 1; x < par->ny-1; x++) {
			if (abs(y - centre_y) % (int) par->sc->scsize[1] == 0) {//only in this row (y value) are reflections
				int offset_x_this_y = offset_x * ((y - centre_y) / (int) par->sc->scsize[1]);
				if (abs(x - (centre_x - offset_x_this_y)) % (int) par->sc->scsize[0] == 0) {//for this row, there are only reflection in these columns (x)
					float value = par->hkl[y*par->ny + x] - ((par->hkl[y*par->ny + x+1] + par->hkl[y*par->ny + x-1] + par->hkl[(y+1)*par->ny + x] + par->hkl[(y-1)*par->ny + x]) / (float) 4);
					if (value < 0){value = 0;}
					fprintf(outfile, "%i\t%i\t%f\n",((x - (centre_x - offset_x_this_y)) / (int) par->sc->scsize[0]) , ((y - centre_y)/(int) par->sc->scsize[1]), value);
				}
			}
		}
	}
	fclose(outfile);
}

//Outputs the exit wave as a list of pixel values 
void pattern_print(struct_parameters *par, int layer) {			
	string out = par->output + ".hkl";
	cout << "Writing hkl output as:\n  " << out << endl;
	FILE *outfile;
	outfile = fopen(out.data(), "w+");
	
	for (int i=0; i < par->ny; i++) {
		for (int j=0; j < par->nx; j++ ) {
			fprintf(outfile, "%f\n", par->hkl[i*par->ny + j]);
		}
	}
	fclose(outfile);
}

#endif