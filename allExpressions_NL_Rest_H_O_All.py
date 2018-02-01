from math import *

def funcNL_Rest_IK82(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41,v42,v43,v44,v45,v46,v47,v48,v49): return abs( ( 0.8825774417257861 - (exp( ( v27 * (cos( v35 )/cos( v28 )) ) )*abs( (abs( v9 )*tan(cos(exp( v35 )))) )) ) )

funcs_IK82 = [funcNL_Rest_IK82,]


funcs = [funcs_IK82,]

def getFuncs(): return funcs
