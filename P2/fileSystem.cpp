/*
 * Autores: Aron Collados (626558)
 *          Cristian Roman (646564)
 */
#include <vector>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <sstream>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <fstream>

using namespace cv;
using namespace std;

struct object{
	string name;
	int num;
	vector<float>mean;
	vector<float>dsv;
};

string filename="data.yml";
string filenameAux="data.yml.bak";
vector<vector<float> > getMoments(const char* obj){
    vector<vector<float> > out;
    ifstream inf (filename.c_str());
    while (inf)
    {
        string strInput,name;
        inf >> strInput;
   	stringstream ss(strInput);
	getline(ss,name, ';');
	if(name==obj){
		while(getline(ss,name, ';')){
			vector<float> v;
			stringstream vs(name);
			while(getline(vs,name, ',')){
				v.push_back(atof(name.c_str()));
			}
			out.push_back(v);
		}
		return out;
	}

    }
    return out;
}
object getObject(const char* name){
	object obj;
	obj.name=name;
	vector<vector<float> > ms=getMoments(name);

	vector<float> mean,dsv,n;
	// Media
	for (int i = 0; i < ms.size(); i++) {
    		vector<float> vf=ms.at(i);
		float f=0,d=0;
    		for (int o = 0; o < vf.size(); o++) {
    			mean.at(o)=mean.at(o)+vf.at(o);
			if(n.size()<o){n.push_back(0);}
			n.at(o)=n.at(o)+1;
		}
		mean.push_back(f);
   	}
	for (int o = 0; o < mean.size(); o++) {	
		mean.at(o)=mean.at(o)/n.at(o);
	}
	// Varianza
	for (int i = 0; i < ms.size(); i++) {
    		vector<float> vf=ms.at(i);
		for (int o = 0; o < vf.size(); o++) {
    			dsv.at(o)=dsv.at(o)+(vf.at(o)-mean.at(o))*(vf.at(o)-mean.at(o));
		}
   	}
	for (int o = 0; o < dsv.size(); o++) {	
		dsv.at(o)=dsv.at(o)/n.at(o);
	}
}
string getMomentData(Moments m){
	std::stringstream d;
	float m0=m.m00;
	float m1=m.m20+m.m02;
	float m2=(m.m20-m.m12)*(m.m20-m.m12)+4*m.m11*m.m11;
	float m3=(m.m30-3*m.m12)*(m.m30-3*m.m12)+
			(3*m.m21-m.m03)*(3*m.m21-m.m03);
	float m4=(m.m30-m.m12)*(m.m30-m.m12)+
			(m.m21+m.m03)*(m.m21+m.m03);
	d<<m0<<","<<m1<<","<<m2<<","<<m3<<","<<m4<<";";
	return d.str();
}
bool addMoment(const char* obj,Moments m){
    ofstream outf(filenameAux.c_str() );
    ifstream inf (filename.c_str());
    bool write=false;
    while (inf)
    {
        string strInput,name;
        inf >> strInput;
   	stringstream ss(strInput);
	getline(ss,name, ';');
	if(name==obj){
		ss<<strInput<<getMomentData(m);
		write=true;
	}
	outf<<ss.str()<<"\n";
    }
    if(!write)
    {
    	outf<<obj<<";"<<getMomentData(m)<<"\n";
    }	
    inf.close();
    outf.close();
    remove(filename.c_str());
    rename(filenameAux.c_str(),filename.c_str());
}
vector<char*> getObjetsTypes(){
    vector<char*> objs;
    ifstream inf (filename.c_str());
    while (inf)
    {
        string strInput,name;
        inf >> strInput;
   	stringstream ss(strInput);
	getline(ss,name, ';');
	char *c;
	strcpy(c, name.c_str());
	objs.push_back(c);
    }
    return objs;
}
vector<object> getObjets(){
	vector<object> objs;
	vector<char*> str=getObjetsTypes();
	for (int i = 0; i < str.size(); i++) {
		objs.push_back(getObject(str.at(i)));
	}
	return objs;
}
