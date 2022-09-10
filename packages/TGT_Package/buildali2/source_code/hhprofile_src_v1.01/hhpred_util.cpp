#include "hhpred_util.h"


/////////////////////////////////////////////////////////////////////////////////////
// fast log base 2
/////////////////////////////////////////////////////////////////////////////////////

// Fast log2 
// ATTENTION: need to compile with g++ -fno-strict-aliasing when using -O2 or -O3!!! 
// Maximum deviation: +/- 2.1E-5
// Run time: ~1.2E-8s on Intel core2 2.13GHz, log2(): 5.4E-8s
// For a negative argument, -128 is returned.
// The function makes use of the representation of 4-byte floating point numbers:
// seee eeee emmm mmmm mmmm mmmm mmmm mmmm
// s is the sign, eee eee e gives the exponent + 127 (in hex: 0x7f).
// The following 23 bits give the mantisse, the binary digits after the decimal 
// point:  x = (-1)^s * 1.mmmmmmmmmmmmmmmmmmmmmmm * 2^(eeeeeeee-127)
// Therefore,  log2(x) = eeeeeeee-127 + log2(1.mmmmmm...) 
//                     = eeeeeeee-127 + log2(1+y),  where y = 0.mmmmmm... 
//                     ~ eeeeeeee-127 + ((a*y+b)*y+c)*y 
// The coefficients a, b  were determined by a least squares fit, and c=1-a-b to get 1 at y=1.
// Lower/higher order polynomials may be used for faster or more precise calculation:
// Order 1: log2(1+y) ~ y                 
// Order 2: log2(1+y) = (a*y + 1-a)*y, a=-0.3427                                           
//  => max dev = +/- 8E-3, run time ~ ?
// Order 3: log2(1+y) = ((a*y+b)*y + 1-a-b)*y, a=0.1564, b=-0.5773   
//  => max dev = +/- 1E-3, run time ~ ?
// Order 4: log2(1+y) = (((a*y+b)*y+c)*y + 1-a-b-c)*y, a=-0.0803 b=0.3170 c=-0.6748 
//  => max dev = +/- 1.4E-4, run time ~ ?
// Order 5: log2(1+y) = ((((a*y+b)*y+c)*y+d)*y + 1-a-b-c-d)*y, 
//     a=0.0440047 b=-0.1903190 c=0.4123442 d=-0.7077702 1-a-b-c-d=1.441740
//  => max dev = +/- 2.1E-5, run time ~ 1.2E-8s
float flog2(float x)
{
  if (x<=0) return -128;
  int *px = (int*)(&x);                 // store address of float as pointer to long int
  float e = (float) (((*px & 0x7F800000) >>23 )-0x7f); // shift right by 23 bits and subtract 127 = 0x7f => exponent
  *px =  ((*px & 0x007FFFFF) | 0x3f800000);  // set exponent to 127 (i.e., 0) 
  x -= 1.0;         // and calculate x-1.0 
  x *= (1.441740 + x*(-0.7077702 +x*(0.4123442 +x*(-0.1903190+x*0.0440047)))); // 5'th order polynomial approx. of log(1+x)
  return x+e;
}

// Fast SSE2 log2 for four floats
// Calculate integer of log2 for four floats in parallel with SSE2
// Maximum deviation: +/- 2.1E-5
// Run time: ~5.6ns on Intel core2 2.13GHz.
// For a negative argument, nonsense is returned. Otherwise, when <1E-38, a value 
// close to -126 is returned and when >1.7E38, +128 is returned.
// The function makes use of the representation of 4-byte floating point numbers:
// seee eeee emmm mmmm mmmm mmmm mmmm mmmm
// s is the sign, eee eee e gives the exponent + 127 (in hex: 0x7f).
// The following 23 bits give the mantisse, the binary digits after the decimal 
// point:  x = (-1)^s * 1.mmmmmmmmmmmmmmmmmmmmmmm * 2^(eeeeeeee-127)
// Therefore,  log2(x) = eeeeeeee-127 + log2(1.mmmmmm...) 
//                     = eeeeeeee-127 + log2(1+y),  where y = 0.mmmmmm... 
//                     ~ eeeeeeee-127 + ((a*y+b)*y+c)*y 
// The coefficients a, b  were determined by a least squares fit, and c=1-a-b to get 1 at y=1.
// Lower/higher order polynomials may be used for faster or more precise calculation:
// Order 1: log2(1+y) ~ y                 
// Order 2: log2(1+y) = (a*y + 1-a)*y, a=-0.3427                                           
//  => max dev = +/- 8E-3, run time ~ 3.8ns
// Order 3: log2(1+y) = ((a*y+b)*y + 1-a-b)*y, a=0.1564, b=-0.5773   
//  => max dev = +/- 1E-3, run time ~ 4.4ns
// Order 4: log2(1+y) = (((a*y+b)*y+c)*y + 1-a-b-c)*y, a=-0.0803 b=0.3170 c=-0.6748 
//  => max dev = +/- 1.4E-4, run time ~ 5.0ns?
// Order 5: log2(1+y) = ((((a*y+b)*y+c)*y+d)*y + 1-a-b-c-d)*y, a=0.0440047 b=-0.1903190 c=0.4123442 d=-0.7077702
//  => max dev = +/- 2.1E-5, run time ~ 5.6ns?

/*
#ifdef HH_SSE2
__m128 _mm_flog2_ps(__m128 X)
{
  const __m128i CONST32_0x7f = _mm_set_epi32(0x7f,0x7f,0x7f,0x7f);
  const __m128i CONST32_0x7fffff = _mm_set_epi32(0x7fffff,0x7fffff,0x7fffff,0x7fffff);
  const __m128i CONST32_0x3f800000 = _mm_set_epi32(0x3f800000,0x3f800000,0x3f800000,0x3f800000);
  const __m128  CONST32_1f = _mm_set_ps(1.0,1.0,1.0,1.0);
  // const float a=0.1564, b=-0.5773, c=1.0-a-b;  // third order
  const float a=0.0440047, b=-0.1903190, c=0.4123442, d=-0.7077702, e=1.0-a-b-c-d; // fifth order
  const __m128  CONST32_A = _mm_set_ps(a,a,a,a);
  const __m128  CONST32_B = _mm_set_ps(b,b,b,b);
  const __m128  CONST32_C = _mm_set_ps(c,c,c,c);
  const __m128  CONST32_D = _mm_set_ps(d,d,d,d);
  const __m128  CONST32_E = _mm_set_ps(e,e,e,e);
  __m128i E; // exponents of X
  __m128 R; //  result
  
  E = _mm_srli_epi32((__m128i) X, 23);    // shift right by 23 bits to obtain exponent+127
  E = _mm_sub_epi32(E, CONST32_0x7f);     // subtract 127 = 0x7f 
  X = (__m128) _mm_and_si128((__m128i) X, CONST32_0x7fffff);  // mask out exponent => mantisse
  X = (__m128) _mm_or_si128((__m128i) X, CONST32_0x3f800000); // set exponent to 127 (i.e., 0) 
  X = _mm_sub_ps(X, CONST32_1f);          // subtract one from mantisse
  R = _mm_mul_ps(X, CONST32_A);           // R = a*X
  R = _mm_add_ps(R, CONST32_B);           // R = a*X+b
  R = _mm_mul_ps(R, X);                   // R = (a*X+b)*X 
  R = _mm_add_ps(R, CONST32_C);           // R = (a*X+b)*X+c
  R = _mm_mul_ps(R, X);                   // R = ((a*X+b)*X+c)*X
  R = _mm_add_ps(R, CONST32_D);           // R = ((a*X+b)*X+c)*X+d
  R = _mm_mul_ps(R, X);                   // R = (((a*X+b)*X+c)*X+d)*X
  R = _mm_add_ps(R, CONST32_E);           // R = (((a*X+b)*X+c)*X+d)*X+e
  R = _mm_mul_ps(R, X);                   // R = ((((a*X+b)*X+c)*X+d)*X+e)*X ~ log2(1+X) !! 
  R = _mm_add_ps(R, _mm_cvtepi32_ps(E));  // convert integer exponent to float and add to mantisse
  return R;
}
#endif
*/

// This function returns log2 with a max absolute deviation of +/- 1.5E-5 (typically 0.8E-5).
// It takes 0.80E-8 s  whereas log2(x) takes 5.4E-7 s. It is hence 9.4 times faster.
// It makes use of the representation of 4-byte floating point numbers:
// seee eeee emmm mmmm mmmm mmmm mmmm mmmm
// s is the sign,
// the following 8 bits, eee eee e, give the exponent + 127 (in hex: 0x7f).
// The following 23 bits give the mantisse, the binary digits after the decimal 
// point:  x = (-1)^s * 1.mmmmmmmmmmmmmmmmmmmmmmm * 2^(eeeeeeee-127)
// In the code, *(int *)&x is an integer which contains the bytes as the 
// floating point variable x is represented in memory. The expression 
//     (((*(int *)&x) & 0x7f800000 ) >>23 )-0x7f is the exponent eeeeeeee, 
// i.e., the largest integer that is smaller than log2(x) (e.g. -1 for 0.9).
float fast_log2(float x)
{
  static float lg2[1025];         // lg2[i] = log2[1+x/1024]
  static float diff[1025];        // diff[i]= (lg2[i+1]-lg2[i])/8096 (for interpolation)
  static char initialized=0;
  if (x<=0) return -100000; 
  if (!initialized)   //First fill in the arrays lg2[i] and diff[i]
    {
      float prev = 0.0f;
      lg2[0] = 0.0f;
      for (int i=1; i<=1024; ++i)
        {
          lg2[i] = log(float(1024+i))*1.442695041-10.0f;
          diff[i-1] = (lg2[i]-prev)*1.2352E-4;
          prev = lg2[i];
        }
      initialized=1;
    }
  int a = (((*((int *)&x)) & 0x7F800000) >>23 )-0x7f; // exponent
  int b =  ((*((int *)&x)) & 0x007FE000) >>13; // first 10 bits of mantisse
  int c =  ((*((int *)&x)) & 0x00001FFF);      // further 13 bits of mantisse
  return a + lg2[b] + diff[b]*(float)(c);
}

// This function returns log gamma and uses fast lookup table for computation.
// This function returns log gamma with a max abolute deviation of +/- 1E-5.
// It makes use of the representation of 4-byte floating point numbers:
// seee eeee emmm mmmm mmmm mmmm mmmm mmmm
// s is the sign,
// the following 8 bits, eee eee e, give the exponent + 127 (in hex: 0x7f).
// The following 23 bits, m, give the mantisse, the binary digits behind the decimal point.
// In summary: x = (-1)^s * 1.mmmmmmmmmmmmmmmmmmmmmm * 2^(eeeeeee-127)
// The expression (((*(int *)&x) & 0x7f800000 ) >>23 )-0x7f is the exponent eeeeeeee, i.e.
// the largest integer that is smaller than log2(x) (e.g. -1 for 0.9). *(int *)&x is an integer which
// contains the bytes as the floating point variable x is represented in memory.
// Check:  assert( sizeof(f) == sizeof(int) );
// Check:  assert( sizeof(f) == 4 );
float fast_log_gamma(float x)
{
  static float log_gamma[1025];   // log_gamma[i] = lgammaf(1+x/1024)
  static float diff[1025];        // diff[i]= (lg2[i+1]-lg2[i])/8096 (for interpolation)
  static char initialized=0;

  if (!initialized) {  //First fill in the arrays log_gamma[i] and diff[i]
      assert(x>=1.0);
      assert( sizeof(x) == sizeof(int) );
      assert( sizeof(x) == 4 );

      float prev = 0.0f;
      log_gamma[0] = 0.0f;
      for (int i=1; i<=1024; ++i) {
          log_gamma[i] = lgammaf(1.0f + i*9.765625E-4);
          diff[i-1] = (log_gamma[i]-prev)*1.2352E-4;
          prev = log_gamma[i];
      }
      initialized=1;
  }

  float res = 1.0f;
  float lres = 0.0f;
  while (x>=2.0f) {
      x -= 1.0;
      res *= x;
      if (res>1E30) {
          lres += fast_log2(res);
          res = 1.0f;
      }
  }
  int b =  ((*((int *)&x)) & 0x007FE000) >>13; // exponent must be 0
  int c =  ((*((int *)&x)) & 0x00001FFF);
  return (lres + fast_log2(res) ) * 0.6931471806 + log_gamma[b] + diff[b]*(float)(c);
}

/////////////////////////////////////////////////////////////////////////////////////
// fast 2^x
// ATTENTION: need to compile with g++ -fno-strict-aliasing when using -O2 or -O3!!!
// Relative deviation < 4.6E-6  (< 2.3E-7 with 5'th order polynomial)
// Speed: 2.1E-8s (2.3E-8s) per call! (exp(): 8.5E-8, pow(): 1.7E-7)
// Internal representation of float number according to IEEE 754: 
//   1bit sign, 8 bits exponent, 23 bits mantissa: seee eeee emmm mmmm mmmm mmmm mmmm mmmm
//                                    0x4b400000 = 0100 1011 0100 0000 0000 0000 0000 0000 
//   In summary: x = (-1)^s * 1.mmmmmmmmmmmmmmmmmmmmmm * 2^(eeeeeee-127)
/////////////////////////////////////////////////////////////////////////////////////
float fpow2(float x)
{
    if (x>FLT_MAX_EXP) return FLT_MAX;
    if (x<FLT_MIN_EXP) return 0.0f;
    int *px = (int*)(&x);                 // store address of float as pointer to long int
    float tx = (x-0.5f) + (3<<22);        // temporary value for truncation: x-0.5 is added to a large integer (3<<22),
                                          // 3<<22 = (1.1bin)*2^23 = (1.1bin)*2^(150-127), 
                                          // which, in internal bits, is written 0x4b400000 (since 10010110bin = 150)
    int lx = *((int*)&tx) - 0x4b400000;   // integer value of x 
    float dx = x-(float)(lx);             // float remainder of x
//   x = 1.0f + dx*(0.69606564f           // cubic apporoximation of 2^x for x in the range [0, 1]
//            + dx*(0.22449433f           // Gives relative deviation < 1.5E-4
//            + dx*(0.07944023f)));       // Speed: 1.9E-8s
    x = 1.0f + dx*(0.693019f             // polynomial approximation of 2^x for x in the range [0, 1]
             + dx*(0.241404f             // Gives relative deviation < 4.6E-6 
             + dx*(0.0520749f            // Speed: 2.1E-8s
             + dx* 0.0134929f )));
//   x = 1.0f + dx*(0.693153f             // polynomial apporoximation of 2^x for x in the range [0, 1]
//            + dx*(0.240153f             // Gives relative deviation < 2.3E-7 
//            + dx*(0.0558282f            // Speed: 2.3E-8s
//            + dx*(0.00898898f
//            + dx* 0.00187682f ))));
    *px += (lx<<23);                      // add integer power of 2 to exponent
    return x;
}



// Normalize a double array such that it sums to one
double normalize_to_one(double* array, size_t length, const float* def_array)
{
    double sum=0.0;
    for (size_t k=0; k<length; ++k) sum+=array[k];
    if (sum!=0.0) {
        double fac=1.0/sum;
        for (size_t k=0; k<length; ++k) array[k]*=fac;
    } else if (def_array) {
        for (size_t k=0; k<length; ++k) array[k]=def_array[k];
    }
    return sum;
}

// Normalize a float array such that it sums to one
float normalize_to_one(float* array, size_t length, const float* def_array)
{
    float sum=0.0;
    for (size_t k=0; k<length; k++) sum+=array[k];
    if (sum!=0.0) {
        float fac=1.0/sum;
        for (size_t k=0; k<length; k++) array[k]*=fac;
    } else if (def_array) {
        for (size_t k=0; k<length; ++k) array[k]=def_array[k];
    }
    return sum;
}

/*
// Check if given filename is a regular file
bool is_regular_file(const char *fn)
{
    bool ret=false;
    struct stat fstats;
    if( stat(fn, &fstats )==0 )
        if( S_ISREG(fstats.st_mode)) ret=true;

    return ret;
}


// Check if given path is a directory
bool is_directory(const char *fn)
{
    bool ret=false;
    struct stat fstats;
    if( stat(fn, &fstats )==0 )
        if( S_ISDIR(fstats.st_mode)) ret=true;

    return ret;
}
*/

// Normalize a float array such that it sums to one
// If it sums to 0 then assign def_array elements to array (optional)
float NormalizeTo1(float* array, int length, float* def_array)
{
  float sum=0.0f;
  int k;
  for (k=0; k<length; k++) sum+=array[k];
  if (sum!=0.0f)
    {
      float fac=1.0/sum;
      for (k=0; k<length; k++) array[k]*=fac;
    }
  else if (def_array)
    for (k=0; k<length; k++) array[k]=def_array[k];
  return sum;
}

// Normalize a float array such that it sums to x
// If it sums to 0 then assign def_array elements to array (optional)
float NormalizeToX(float* array, int length, float x, float* def_array)
{
  float sum=0.0;
  int k;
  for (k=0; k<length; k++) sum+=array[k];
  if (sum)
    {
      float fac=x/sum;
      for (k=0; k<length; k++) array[k]*=fac;
    }
  else if (def_array)
    for (k=0; k<length; k++) array[k]=def_array[k];
  return sum;
}

/////////////////////////////////////////////////////////////////////////////////////
// Similar to spintf("%*g,w,val), but displays maximum number of digits within width w
/////////////////////////////////////////////////////////////////////////////////////
char* sprintg(float val, int w)
{
  static char str[100];
  float log10val = log10(fabs(val));
  int neg = (val<0? 1: 0);
  if (log10val >= w-neg-1 || -log10val > 3)
    {
      // positive exponential 1.234E+06
      // negative exponential 1.234E-06
      int d = w-6-neg;
      sprintf(str,"%*.*e",w,d<1?1:d,val);
    }
  else
    {
      int d = log10val>0? w-2-neg-int(log10val): w-2-neg;
      sprintf(str,"%#*.*f",w,d,val);
    }
  return str;
}

/////////////////////////////////////////////////////////////////////////////////////
// String utilities
/////////////////////////////////////////////////////////////////////////////////////

// Safe strcpy command that limits copy to a maximum of maxlen+1 characters 
// including the '\0' character.
// (Different from standard strncpy, which pads with \0 characters)
// Returns maxlen minus the number of non-\0 characters copied
// If the string is cut prematurely it will return 0! 
int strmcpy(char* dest, const char* source, size_t maxlen)
{
  while (*source && (maxlen--)>0) *dest++ = *source++; 
  *dest = '\0';
  return maxlen;
}

// Scans string for integer number starting from ptr and returns this number.
// The ptr will then be moved to the first position after the integer number.
// If no integer number is found, it returns INT_MIN and sets ptr to NULL.
int strtoi(const char*& ptr)
{
    int i;
    const char* ptr0=ptr;
    if (!ptr) return INT_MIN;
    while (*ptr!='\0' && !(*ptr>='0' && *ptr<='9')) ptr++;
    if (*ptr=='\0') {
        ptr=0;
        return INT_MIN;
    }
    if (*(ptr-1)=='-' && ptr>ptr0) i=-atoi(ptr); else i=atoi(ptr);
    while (*ptr>='0' && *ptr<='9') ptr++;
    return i;
}


//Same as strint, but interpretes '*' as default
int strtoi_(const char*& ptr, int deflt)
{
    int i;
    if (!ptr) return INT_MIN;
    while (*ptr!='\0' && !(*ptr>='0' && *ptr<='9') && *ptr!='*') ptr++;
    if (*ptr=='\0') {
        ptr=0;
        return INT_MIN;
    }
    if (*ptr=='*') {
        ptr++;
        return deflt;
    }
    if (*(ptr-1)=='-') i=atoi(ptr-1);
    else i=atoi(ptr);
    while (*ptr>='0' && *ptr<='9') ptr++;
    return i;
}


// Returns leftmost integer in ptr and sets the pointer to first char after
// the integer. If no integer is found, returns INT_MIN and sets pt to NULL
int strint(char*& ptr)
{
  int i;
  char* ptr0=ptr;
  if (!ptr) return INT_MIN;
  while (*ptr!='\0' && !(*ptr>='0' && *ptr<='9')) ptr++;
  if (*ptr=='\0')
    {
      ptr=0;
      return INT_MIN;
    }
  if (ptr>ptr0 && *(ptr-1)=='-') i=-atoi(ptr); else i=atoi(ptr);
  while (*ptr>='0' && *ptr<='9') ptr++;
  return i;
}

// Same as strint, but interpretes '*' as default
int strinta(char*& ptr, int deflt)
{
  int i;
  if (!ptr) return INT_MIN;
  while (*ptr!='\0' && !(*ptr>='0' && *ptr<='9') && *ptr!='*') ptr++;
  if (*ptr=='\0')
    {
      ptr=0;
      return INT_MIN;
    }
  if (*ptr=='*')
    {
      ptr++;
      return deflt;
    }
  if (*(ptr-1)=='-') i=atoi(ptr-1);
  else i=atoi(ptr);
  while (*ptr>='0' && *ptr<='9') ptr++;
  return i;
}

// Returns leftmost float in ptr and sets the pointer to first char after
// the float. If no float is found, returns FLT_MIN and sets pt to NULL
float strflt(char*& ptr)
{
  float i;
  char* ptr0=ptr;
  if (!ptr) return FLT_MIN;
  while (*ptr!='\0' && !(*ptr>='0' && *ptr<='9')) ptr++;
  if (*ptr=='\0')
    {
      ptr=0;
      return FLT_MIN;
    }
  if (ptr>ptr0 && *(ptr-1)=='-') i=-atof(ptr); else i=atof(ptr);
  while ((*ptr>='0' && *ptr<='9') || *ptr=='.') ptr++;
  return i;
}

// Same as strint, but interpretes '*' as default
float strflta(char*& ptr, float deflt)
{
  float i;
  if (!ptr) return FLT_MIN;
  while (*ptr!='\0' && !(*ptr>='0' && *ptr<='9') && *ptr!='*') ptr++;
  if (*ptr=='\0')
    {
      ptr=0;
      return FLT_MIN;
    }
  if (*ptr=='*')
    {
      ptr++;
      return deflt;
    }
  if (*(ptr-1)=='-') i=-atof(ptr);
  else i=atof(ptr);
  while ((*ptr>='0' && *ptr<='9') || *ptr=='.') ptr++;
  return i;
}


// Removes the newline and other control characters at the end of a string (if present)
// and returns the new length of the string (-1 if str is NULL)
int chomp(char str[])
{
  if (!str) return -1;
  int l=0;
  for (l=strlen(str)-1; l>=0 && str[l]<32; l--);
  str[++l]='\0';
  return l;
}

// Emulates the ifstream::getline method; similar to fgets(str,maxlen,FILE*),
// but removes the newline at the end and returns NULL if at end of file or read error
char* fgetline(char str[], const int maxlen, FILE* file)
{
  if (!fgets(str,maxlen,file)) return NULL;
  if (chomp(str)+1>=maxlen)    // if line is cut after maxlen characters...
    while (fgetc(file)!='\n'); // ... read in rest of line
  
  return(str);
}

// copies substring str[a,b] into substr and returns substr
char *substr(char* substr, char* str, int a, int b)
{
  if (b<a) {int i=b; b=a; a=i;}
  if (b-a>1000)
    {printf("Function substr: >1000 chars to copy. Exiting.\n"); exit(6);}
  char* dest=substr;
  char* source=str+a;
  char* send=str+b;
  while (*source!='\0' && source<=send) *(dest++) = *(source++);
  *dest='\0';
  return substr;
}


// Returns pointer to first non-white-space character in str OR to NULL if none found
char* strscn(char* str)
{
  if (!str) return NULL;
  char* ptr=str;
  while (*ptr!='\0' && *ptr<=32) ptr++;
  return (*ptr=='\0')? NULL: ptr;
}

// Returns pointer to first white-space character in str OR to NULL if none found
char* strscn_ws(char* str)
{
  if (!str) return NULL;
  char* ptr=str;
  while (*ptr!='\0' && *ptr>32) ptr++;
  return (*ptr=='\0')? NULL: ptr;
}

//Returns pointer to first non-white-space character in str OR to NULL if none found
const char* strscn_c(const char* str)
{
    if (!str) return NULL;
    const char* ptr=str;
    while (*ptr!='\0' && isspace(*ptr)) ptr++;
    return (*ptr=='\0') ? NULL : ptr;
}

// Returns pointer to first  non-white-space character in str OR to end of string '\0' if none found
char* strscn_(char* str)
{
  if (!str) return NULL;
  char* ptr=str;
  while (*ptr!='\0' && *ptr<=32) ptr++;
  return ptr;
}

// Returns pointer to first non-c character in str OR to NULL if none found
char* strscn(char* str, const char c)
{
  if (!str) return NULL;
  char* ptr=str;
  while (*ptr!='\0' && *ptr==c) ptr++;
  return (*ptr=='\0')? NULL: ptr;
}

// Returns pointer to first  non-c character in str OR to end of string '\0' if none found
char* strscn_(char* str, const char c)
{
  if (!str) return NULL;
  char* ptr=str;
  while (*ptr!='\0' && *ptr==c) ptr++;
  return ptr;
}

// Cuts string at first white space character found by overwriting it with '\0'.
// Returns pointer to next non-white-space char OR to NULL if no such char found
char* strcut(char* str)
{
  if (!str) return NULL;
  char* ptr=str;
  while (*ptr!='\0' && *ptr>32) ptr++;
  if (*ptr=='\0') return NULL;
  *ptr='\0';
  ptr++;
  while (*ptr!='\0' && *ptr<=32) ptr++;
  return (*ptr=='\0')? NULL:ptr;
}

// Cuts string at first white space character found by overwriting it with '\0'.
// Returns pointer to next non-white-space char OR to end of string '\0' if none found
char* strcut_(char* str)
{
  if (!str) return NULL;
  char* ptr=str;
  while (*ptr!='\0' && *ptr>32) ptr++;
  if (*ptr=='\0') return ptr;
  *ptr='\0';
  ptr++;
  while (*ptr!='\0' && *ptr<=32) ptr++;
  return ptr;
}

// Cuts string at first occurence of charcter c, by overwriting it with '\0'.
// Returns pointer to next char not equal c, OR to NULL if none found
char* strcut(char* str, const char c)
{
  if (!str) return NULL;
  char* ptr=str;
  while (*ptr!='\0' && *ptr!=c) ptr++;
  if (*ptr=='\0') return NULL;
  *ptr='\0';
  ptr++;
  while (*ptr!='\0' && *ptr==c) ptr++;
  return (*ptr=='\0')? NULL:ptr;
}

// Cuts string at first occurence of charcter c, by overwriting it with '\0'.
// Returns pointer to next char not equal c, OR to end of string '\0' if none found
char* strcut_(char* str, const char c)
{
  if (!str) return NULL;
  char* ptr=str;
  while (*ptr!='\0' && *ptr!=c) ptr++;
  if (*ptr=='\0') return ptr;
  *ptr='\0';
  ptr++;
  while (*ptr!='\0' && *ptr==c) ptr++;
  return ptr;
}

// Cuts string at first occurence of substr, by overwriting the first letter with '\0'.
// Returns pointer to next char after occurence of substr, OR to NULL if no such char found
char* strcut(char* str, const char* substr)
{
  char* ptr;     //present location in str being compared to substr
  const char* sptr=substr; //present location in substr being compared to substr
  // while not at end of str and not all of substr is matched yet
  while (1)
    {
      for (ptr=str, sptr=substr; *ptr==*sptr && *ptr!='\0';  ptr++, sptr++) ;
      if (*sptr=='\0') {*str='\0'; return ptr;}
      if (*ptr=='\0')  return NULL;
      str++;
    }
}

// Cuts string at first occurence of substr, by overwriting the first letter with '\0'.
// Returns pointer to next char after occurence of substr, OR to end of string '\0' if no such char found
char* strcut_(char* str, const char* substr)
{
  char* ptr;         //present location in str being compared to substr
  const char* sptr=substr; //present location in substr being compared to str
  // while not at end of str and not all of substr is matched yet
  while (1)
    {
      for (ptr=str, sptr=substr; *ptr==*sptr && *ptr!='\0';  ptr++, sptr++) ;
      if (*sptr=='\0') {*str='\0'; return ptr;}
      if (*ptr=='\0')  return ptr;
      str++;
    }
}

// Copies first word in ptr to str. In other words, copies first block of non whitespace characters,
// beginning at ptr, to str. If a word is found, returns address of second word in ptr or, if no second
// word is found, returns address to end of word ('\0' character) in ptr string. If no word is found
// in ptr NULL is returned.
char* strwrd(char* str, char* ptr)
{
  ptr=strscn(ptr);    // advance to beginning of next word
  if (ptr)
    {
      while (*ptr!='\0' && *ptr>32) *(str++) = *(ptr++);
      *str='\0';
      while (*ptr!='\0' && *ptr<=32) ptr++;
      return ptr;
    }
  else return NULL;
}

// Copies first word in ptr to str. In other words, copies first block of non whitespace characters,
// beginning at ptr, to str. If a word is found, returns address of second word in ptr or, if no second
// word is found, returns address to end of word ('\0' character) in ptr string. If no word is found
// in ptr NULL is returned.
char* strwrd(char* str, char* ptr, int maxlen)
{
  ptr=strscn(ptr);    // advance to beginning of next word
  if (ptr)
    {
      while (*ptr!='\0' && *ptr>32 && (maxlen--)>0) *(str++) = *(ptr++);
      *str='\0';
      while (*ptr!='\0' && *ptr<=32) ptr++;
      return ptr;
    }
  else return NULL;
}

// Copies first word ***delimited by char c*** in ptr to str. In other words, copies first block of non-c characters,
// beginning at ptr, to str. If a word is found, returns address of second word in ptr or, if no second
// word is found, returns address to end of word ('\0' character) in ptr string. If no word is found
// in ptr NULL is returned.
char* strwrd(char* str, char* ptr, const char c)
{
  ptr=strscn(ptr,c);    // advance to beginning of next word
  if (ptr)
    {
      while (*ptr!='\0' && *ptr!=c) *(str++) = *(ptr++);
      *str='\0';
      while (*ptr!='\0' && *ptr==c) ptr++;
     return ptr;
    }
  else return NULL;
}

// Similar to Perl's tr/abc/ABC/: Replaces all chars in str found in one list with characters from the second list
// Returns the number of replaced characters
int strtr(char* str, const char oldchars[], const char newchars[])
{
  char* ptr;
  const char *plist;
  int ntr=0;
  for (ptr=str; *ptr!='\0'; ptr++)
    for (plist=oldchars; *plist!='\0'; plist++)
      if (*ptr==*plist)
        {
          *ptr=newchars[plist-oldchars];
          ntr++;
          break;
        }
  return ntr;
}

// Similar to Perl's tr/abc//d: deletes all chars in str found in the list
// Returns number of removed characters
int strtrd(char* str, const char chars[])
{
  char* ptr0=str;
  char* ptr1=str;
  const char *plist;
  while (*ptr1!='\0')
    {
      for (plist=chars; *plist!='\0'; plist++)
        if (*ptr1==*plist) break;
      if (*plist=='\0') {*ptr0=*ptr1; ptr0++;}
      ptr1++;
    }
  *ptr0=*ptr1;
  return ptr1-ptr0;
}

// Similar to Perl's tr/a-z//d: deletes all chars in str found in the list
// Returns number of removed characters
int strtrd(char* str, char char1, char char2)
{
  char* ptr0=str;
  char* ptr1=str;
  while (*ptr1!='\0')
    {
      if (*ptr1>=char1 && *ptr1<=char2) {*ptr0=*ptr1; ptr0++;}
      ptr1++;
    }
  *ptr0=*ptr1;
  return ptr1-ptr0;
}

// Counts the number of characters in str that are in range between char1 and char2
int strcount(char* str, char char1, char char2)
{
  char* ptr=str;
  int count=0;
  while (*ptr!='\0')
    {
      if (*ptr>=char1 && *ptr<=char2) count++;
      ptr++;
    }
  return count;
}

// transforms str into an all uppercase string
char* uprstr(char* str)
{
  char* s=str;
  while (*s !='\0') {if (*s>='a' && *s<='z') *s+='A'-'a';s++;}
  return(str);
}

// transforms str into an all uppercase string
char* lwrstr(char* str)
{
  char* s=str;
  while (*s !='\0') {if (*s>='A' && *s<='Z') *s+='a'-'A'; s++;}
  return(str);
}

// transforms chr into an uppercase character
char uprchr(char chr)
{
  return (chr>='a' && chr<='z')? chr+'A'-'a' : chr;
}

// transforms chr into an lowercase character
char lwrchr(char chr)
{
  return (chr>='A' && chr<='Z')? chr-'A'+'a' : chr;
}


// Replaces first occurence of str1 by str2 in str. Returns pointer to first occurence or NULL if not found
// ATTENTION: if str2 is longer than str1, allocated memory of str must be long enough!!
char* strsubst(char* str, const char str1[], const char str2[])
{
  char* ptr = strstr(str,str1);
  strcpy(ptr,str2);
  return ptr;
}

/*
// Gives elapsed time since first call to this function
void ElapsedTimeSinceFirstCall(const char str[])
{
  timeval t;
  static double tfirst=0;
  if (tfirst==0)
    {
      gettimeofday(&t, NULL);
      tfirst = 1E-6*t.tv_usec + t.tv_sec;
    }
  gettimeofday(&t, NULL);
  printf("Elapsed time since first call:%12.3fs %s\n",1E-6*t.tv_usec + t.tv_sec - tfirst,str);
}

// Gives elapsed time since last call to this function
void ElapsedTimeSinceLastCall(const char str[])
{
  timeval t;
  static double tlast=0.0;
  if (tlast==0.0)
    {
      gettimeofday(&t, NULL);
      tlast = 1.0E-6*t.tv_usec + t.tv_sec;
    }
  gettimeofday(&t, NULL);
  printf("Elapsed time since last call:%12.3fs %s\n",1.0E-6*t.tv_usec + t.tv_sec - tlast,str);
  tlast = 1.0E-6*t.tv_usec + t.tv_sec;
}
*/

char* RemovePath(char outname[], char filename[])
{
  char* ptr;
#ifdef WINDOWS
  ptr=strrchr(filename,92);  //return adress for LAST \ (backslash) in name
#else
  ptr=strrchr(filename,'/'); //return adress for LAST / in name
#endif
  if (!ptr) ptr=filename; else ptr++;
  strcpy(outname,ptr);
  return outname;
}

char* RemoveExtension(char outname[], char filename[])
{
  char *ptr1;
  ptr1=strrchr(filename,'.');       //return adress for LAST '.' in name
  if (ptr1) {*ptr1='\0'; strcpy(outname,filename); *ptr1='.';} else strcpy(outname,filename);
  return outname;
}

char* RemovePathAndExtension(char outname[], char filename[])
{
  char *ptr, *ptr1;
#ifdef WINDOWS
  ptr=strrchr(filename,92);  //return adress for LAST \ (backslash) in name
#else
  ptr=strrchr(filename,'/'); //return adress for LAST / in name
#endif
  if (!ptr) ptr=filename; else ptr++;
  ptr1=strrchr(filename,'.');       //return adress for LAST '.' in name
  if (ptr1) {*ptr1='\0'; strcpy(outname,ptr); *ptr1='.';} else strcpy(outname,ptr);
  return outname;
}

char* Extension(char extension[], char filename[])
{
  char* ptr;
  ptr=strrchr(filename,'.');      //return adress for LAST '.' in name
  if (ptr) strcpy(extension,ptr+1); else *extension='\0';
  return extension;
}

// Path includes last '/'
char* Pathname(char pathname[], char filename[])
{
  char* ptr;
  char chr;
#ifdef WINDOWS
  ptr=strrchr(filename,92);  //return adress for LAST \ (backslash) in name
#else
  ptr=strrchr(filename,'/'); //return adress for LAST / in name
#endif
  if (ptr) {chr=*(++ptr); *ptr='\0'; strcpy(pathname,filename); *ptr=chr;} else *pathname='\0';
  return pathname;
}

// Swaps two integer elements in array k
void swapi(int k[], int i, int j)
{
  int temp;
  temp=k[i]; k[i]=k[j]; k[j]=temp;
}

// QSort sorting routine. time complexity of O(N ln(N)) on average
// Sorts the index array k between elements i='left' and i='right' in such a way that afterwards
// v[k[i]] is sorted downwards (up=-1) or upwards (up=+1)
void QSortInt(int v[], int k[], int left, int right, int up)
{
  int i;
  int last;   // last element to have been swapped

  if (left>=right) return;        // do nothing if less then 2 elements to sort
  // Put pivot element in the middle of the sort range to the side (to position 'left') ...
  swapi(k,left,(left+right)/2);
  last=left;
  // ... and swap all elements i SMALLER than the pivot
  // with an element that is LARGER than the pivot (element last+1):
  if (up==1)
    {
      for (i=left+1; i<=right; i++)
        if (v[k[i]]<v[k[left]]) swapi(k,++last,i);
    }
  else
    for (i=left+1; i<=right; i++)
      if (v[k[i]]>v[k[left]]) swapi(k,++last,i);

  // Put the pivot to the right of the elements which are SMALLER, left to elements which are LARGER
  swapi(k,left,last);

  // Sort the elements left from the pivot and right from the pivot
  QSortInt(v,k,left,last-1,up);
  QSortInt(v,k,last+1,right,up);
}

// QSort sorting routine. time complexity of O(N ln(N)) on average
// Sorts the index array k between elements i='left' and i='right' in such a way that afterwards
// v[k[i]] is sorted downwards (up=-1) or upwards (up=+1)
void QSortFloat(float v[], int k[], int left, int right, int up)
{
  int i;
  int last;   // last element to have been swapped
  void swapi(int k[], int i, int j);

  if (left>=right) return;        // do nothing if less then 2 elements to sort
  // Put pivot element in the middle of the sort range to the side (to position 'left') ...
  swapi(k,left,(left+right)/2);
  last=left;
  // ... and swap all elements i SMALLER than the pivot
  // with an element that is LARGER than the pivot (element last+1):
  if (up==1)
    {
    for (i=left+1; i<=right; i++)
      if (v[k[i]]<v[k[left]]) swapi(k,++last,i);
    }
  else
    for (i=left+1; i<=right; i++)
      if (v[k[i]]>v[k[left]]) swapi(k,++last,i);

  // Put the pivot to the right of the elements which are SMALLER, left to elements which are LARGER
  swapi(k,left,last);

  // Sort the elements left from the pivot and right from the pivot
  QSortFloat(v,k,left,last-1,up);
  QSortFloat(v,k,last+1,right,up);
}

// QSort sorting routine. time complexity of O(N ln(N)) on average
// Sorts the index array k between elements i='left' and i='right' in such a way that afterwards
// v[k[i]] is sorted downwards (up=-1) or upwards (up=+1)
void QSortDouble(double v[], int k[], int left, int right, int up)
{
  int i;
  int last;   // last element to have been swapped
  void swapi(int k[], int i, int j);

  if (left>=right) return;        // do nothing if less then 2 elements to sort
  // Put pivot element in the middle of the sort range to the side (to position 'left') ...
  swapi(k,left,(left+right)/2);
  last=left;
  // ... and swap all elements i SMALLER than the pivot
  // with an element that is LARGER than the pivot (element last+1):
  if (up==1)
    {
    for (i=left+1; i<=right; i++)
      if (v[k[i]]<v[k[left]]) swapi(k,++last,i);
    }
  else
    for (i=left+1; i<=right; i++)
      if (v[k[i]]>v[k[left]]) swapi(k,++last,i);

  // Put the pivot to the right of the elements which are SMALLER, left to elements which are LARGER
  swapi(k,left,last);

  // Sort the elements left from the pivot and right from the pivot
  QSortDouble(v,k,left,last-1,up);
  QSortDouble(v,k,last+1,right,up);
}


//============= additional =============//
float fast_dot_product_single2(float* qi, float* tj,int num)
{
#ifdef HH_SSE3
        float __attribute__((aligned(16))) res;
        __m128 P; // query 128bit SSE2 register holding 4 floats
        __m128 R; // result
        __m128* Qi = (__m128*) qi;
        __m128* Tj = (__m128*) tj;
        int iter=(int)(num/4)-2;
//head
        R = _mm_mul_ps(*(Qi++),*(Tj++));  //first 4
//iter
        for(int i=0;i<iter;i++)   //mid 4*
        {
                P = _mm_mul_ps(*(Qi++),*(Tj++)); //1
                R = _mm_add_ps(R,P);  //1
        }
//tail
        P = _mm_mul_ps(*Qi,*Tj);  //last 4
        R = _mm_add_ps(R,P);
        R = _mm_hadd_ps(R,R);
        R = _mm_hadd_ps(R,R);
        _mm_store_ss(&res, R);

//--- final remain ---//
//      int wsremain=(int)(num%4);
//      float ws2=0;
//      for(int i=0;i<wsremain;i++)ws2+=qi[num-1-i]*tj[num-1-i];
//      return res+ws2;
        return res;

#else

        float ws2=0;
        for(int i=0;i<num;i++)ws2+=qi[i]*tj[i];
        return ws2;

#endif
}


// Calculate scalar product between 20 dimensional float vectors qi and tj
float ScalarProd20(float* qi, float* tj)
{

//#ifndef HH_SSE2
  return  tj[0] *qi[0] +tj[1] *qi[1] +tj[2] *qi[2] +tj[3] *qi[3] +tj[4] *qi[4]
    +tj[5] *qi[5] +tj[6] *qi[6] +tj[7] *qi[7] +tj[8] *qi[8] +tj[9] *qi[9]
    +tj[10]*qi[10]+tj[11]*qi[11]+tj[12]*qi[12]+tj[13]*qi[13]+tj[14]*qi[14]
    +tj[15]*qi[15]+tj[16]*qi[16]+tj[17]*qi[17]+tj[18]*qi[18]+tj[19]*qi[19];
//#endif

/*
#ifdef HH_SSE2
  float __attribute__((aligned(16))) res;
  __m128 P; // query 128bit SSE2 register holding 4 floats
  __m128 R; // result  
  __m128* Qi = (__m128*) qi;
  __m128* Tj = (__m128*) tj;

#ifdef HH_SSE41
  R = _mm_dp_ps(*(Qi++),*(Tj++),0xFF); // dot product
  P = _mm_dp_ps(*(Qi++),*(Tj++),0xFF); // dot product
  R = _mm_add_ps(R,P);
  P = _mm_dp_ps(*(Qi++),*(Tj++),0xFF);
  R = _mm_add_ps(R,P);
  P = _mm_dp_ps(*(Qi++),*(Tj++),0xFF);
  R = _mm_add_ps(R,P);
  P = _mm_dp_ps(*Qi,*Tj,0xFF);
  R = _mm_add_ps(R,P);
#else
  R = _mm_mul_ps(*(Qi++),*(Tj++));
  P = _mm_mul_ps(*(Qi++),*(Tj++));
  R = _mm_add_ps(R,P);
  P = _mm_mul_ps(*(Qi++),*(Tj++));
  R = _mm_add_ps(R,P);
  P = _mm_mul_ps(*(Qi++),*(Tj++));
  R = _mm_add_ps(R,P);
  P = _mm_mul_ps(*Qi,*Tj);
  R = _mm_add_ps(R,P);
#ifdef HH_SSE3
  R = _mm_hadd_ps(R,R);
  R = _mm_hadd_ps(R,R);
#else // equivalent SSE2 code
  P = _mm_shuffle_ps(R,R, _MM_SHUFFLE(2,0,2,0));
  R = _mm_shuffle_ps(R,R, _MM_SHUFFLE(3,1,3,1));
  R = _mm_add_ps(R,P);
  P = _mm_shuffle_ps(R,R, _MM_SHUFFLE(2,0,2,0));
  R = _mm_shuffle_ps(R,R, _MM_SHUFFLE(3,1,3,1));
  R = _mm_add_ps(R,P);
#endif // end SSE2 code
#endif // end ifndef HH_SSE41
  _mm_store_ss(&res, R);
  return res;
#endif // end ifdef HH_SSE2
*/
}


//================ errors ===================//
/////////////////////////////////////////////////////////////////////////////////////
// Errors
/////////////////////////////////////////////////////////////////////////////////////
int FormatError(const char infile[], const char details[])
{
  cerr<<"Error : wrong format while reading file \'"<<infile<<". "<<details<<"\n";
  exit(1);
}

int OpenFileError(const char outfile[])
{
  cerr<<endl<<"Error : could not open file \'"<<outfile<<"\'\n";
  exit(2);
}

int MemoryError(const char arrayname[])
{
  cerr<<"Error : Could not allocate memory for \'"<<arrayname<<"\'.\nIs your max memory size sufficient? (Check using '$ ulimit -a' under Linux)"<<endl;
  exit(3);
}

int NoMemoryError(const char arrayname[])
{
  cerr<<"Error : Could not allocate memory for \'"<<arrayname<<"\'.\nIs your max memory size sufficient? (Check using '$ ulimit -a' under Linux)"<<endl; 
  exit(3);
}

int SyntaxError(const char details[])
{
  cerr<<"Error on command line: "<<details<<"\n";
  exit(4);
}

int InternalError(const char errstr[])
{
  cerr<<"Error :  "<<errstr<<". Please report this bug to the developers\n";
  exit(6);
}


