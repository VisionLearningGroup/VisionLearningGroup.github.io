// $Id: getmaxviol.C,v 1.1.1.1 2007/07/24 19:44:11 pjain Exp $
#include "mex.h"
#include <math.h>
//#define INVOCATION_NAME startGdb
//#include "startgdb.h"

// The entry point searched for and called by Matlab.  See also:
// www.mathworks.com/access/helpdesk/help/techdoc/apiref/mexfunction_c.html
void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray* prhs[])
{
	const mxArray *X;
	const mxArray *cnst;
	const mxArray *slack;
	double *Xvals;
	double *cnstvals;
	double *slackvals;
	double *out, *out2;
	int n,c,i;
	int currind = 1;
	double viol, currmaxviol = 0;
	X = prhs[0];
	cnst = prhs[1];
	slack= prhs[2];
	Xvals = mxGetPr(X);
	cnstvals = mxGetPr(cnst);
	slackvals= mxGetPr(slack);
	n = mxGetN(X);
	c = mxGetM(cnst);
	for(i = 0; i < c; i++)
	{
	  viol = Xvals[((int)cnstvals[1*c+i] - 1)*n + (int)cnstvals[1*c+i] - 1] + Xvals[((int)cnstvals[2*c+i] - 1)*n + (int)cnstvals[2*c+i] - 1] - 2*Xvals[((int)cnstvals[1*c+i] - 1)*n + (int)cnstvals[2*c+i] - 1] - slackvals[i];//cnstvals[3*c+i];
		if(cnstvals[4*c+i] == -1)
		  viol = -1*viol;
// 		if(viol < 0)
// 			viol = -1*viol;
		if(viol > currmaxviol)
		{
			currmaxviol = viol;
			currind = i;
		}
	}
	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
	out = mxGetPr(plhs[0]);
	out[0] = currind+1;
	plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
	out2 = mxGetPr(plhs[1]);
	out2[0] = currmaxviol;
    return;
}
