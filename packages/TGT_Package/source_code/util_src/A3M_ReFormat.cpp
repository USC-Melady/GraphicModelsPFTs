#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <string.h>
using namespace std;


//=================== upper and lower case ====================//
//----------upper_case-----------//
void toUpperCase(char *buffer)
{
	for(int i=0;i<(int)strlen(buffer);i++)
	if(buffer[i]>=97 && buffer[i]<=122) buffer[i]-=32;
}
void toUpperCase(string &buffer)
{
	for(int i=0;i<(int)buffer.length();i++)
	if(buffer[i]>=97 && buffer[i]<=122) buffer[i]-=32;
}
//----------lower_case-----------//
void toLowerCase(char *buffer)
{
	for(int i=0;i<(int)strlen(buffer);i++)
	if(buffer[i]>=65 && buffer[i]<=90) buffer[i]+=32;
}
void toLowerCase(string &buffer)
{
	for(int i=0;i<(int)buffer.length();i++)
	if(buffer[i]>=65 && buffer[i]<=90) buffer[i]+=32;
}

//----- get upper case -----//
int getUpperCase(char *buffer)
{
	int count=0;
	for(int i=0;i<(int)strlen(buffer);i++)
	if(buffer[i]>=65 && buffer[i]<=90) count++;
	return count;
}
int getUpperCase(string &buffer)
{
	int count=0;
	for(int i=0;i<(int)buffer.length();i++)
	if(buffer[i]>=65 && buffer[i]<=90) count++;
	return count;
}
//----- get lower case -----//
int getLowerCase(char *buffer)
{
	int count=0;
	for(int i=0;i<(int)strlen(buffer);i++)
	if(buffer[i]>=97 && buffer[i]<=122) count++;
	return count;
}
int getLowerCase(string &buffer)
{
	int count=0;
	for(int i=0;i<(int)buffer.length();i++)
	if(buffer[i]>=97 && buffer[i]<=122) count++;
	return count;
}

//----- trim white space -----//
int Trim_White_Space(string &in)
{
	int count=0;
	string out="";
	for(int i=0;i<(int)in.length();i++)
	if(in[i]!=' ')out+=in[i],count++;
	in=out;
	return count;
}



//-------- read in MSA in a3m format (i.e., normal FASTA with upper/lower) ------------//
//[note]: we set the first sequence as the query sequence,
//        that is to say, all the following sequences should be longer than the first
int Multi_FASTA_Input(string &multi_fasta,vector <string> &nam_list,vector <string> &fasta_list)
{
	ifstream fin;
	string buf,temp;
	//read
	fin.open(multi_fasta.c_str(), ios::in);
	if(fin.fail()!=0)
	{
		fprintf(stderr,"file %s not found!\n",multi_fasta.c_str());
		return -1;
	}
	//load
	int relfirst=1;
	int firstlen;
	int first=1;
	int count=0;
	int number=0;
	string name;
	string seq;
	nam_list.clear();
	fasta_list.clear();
	for(;;)
	{
		if(!getline(fin,buf,'\n'))break;
		if(buf=="")continue;
		if(buf.length()>=1 && buf[0]=='>')
		{
			name=buf.substr(1,buf.length()-1);
			nam_list.push_back(name);
			count++;
			if(first!=1)
			{
				fasta_list.push_back(seq);
				number++;
				if(number==1)
				{
					firstlen=(int)seq.length();
				}
				else
				{
					Trim_White_Space(seq);
					int lowlen=getLowerCase(seq);
					int curlen=(int)seq.length()-lowlen;
					if(curlen!=firstlen)
					{
						fprintf(stderr,"length not equal at %s, [%d!=%d] \n",buf.c_str(),curlen,firstlen);
						return -1;
					}
				}
			}
			first=0;
			seq="";
		}
		else
		{
			if(first!=1)seq+=buf;
		}
	}
	//final
	if(first!=1)
	{
		fasta_list.push_back(seq);
		number++;
		if(number==1)
		{
			firstlen=(int)seq.length();
		}
		else
		{
			Trim_White_Space(seq);
			int lowlen=getLowerCase(seq);
			int curlen=(int)seq.length()-lowlen;
			if(curlen!=firstlen)
			{
				fprintf(stderr,"length not equal at %s, [%d!=%d] \n",buf.c_str(),curlen,firstlen);
				return -1;
			}
		}
	}
	//check
	if(number!=count)
	{
		fprintf(stderr,"num %d != count %d \n",number,count);
		return -1;
	}
	return count;
}


//-------- main -------//
int main(int argc,char **argv)
{
	//------ A3M_ReFormat -------//
	{
		if(argc<3)
		{
			fprintf(stderr,"Version: 0.99 \n");
			fprintf(stderr,"A3M_ReFormat <a3m_input> <a3m_output> \n");
			exit(-1);
		}
		string a3m_input=argv[1];
		string a3m_output=argv[2];
		vector <string> nam_list;
		vector <string> fasta_list;
		int totnum=Multi_FASTA_Input(a3m_input,nam_list,fasta_list);
		if(totnum<=0)exit(-1);
		//output
		FILE *fp=fopen(a3m_output.c_str(),"wb");
		for(int i=0;i<totnum;i++)
		{
			fprintf(fp,">%s\n",nam_list[i].c_str());
			fprintf(fp,"%s\n",fasta_list[i].c_str());
		}
		fclose(fp);
		//exit
		exit(0);
	}
}

