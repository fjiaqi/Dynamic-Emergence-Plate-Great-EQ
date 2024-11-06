import underworld as uw
import numpy as np
from mpi4py import MPI

def getGlobalSwarmVar(swarmdata):
    if(uw.mpi.size == 1):
        return swarmdata
    datalenmax = np.array(0, 'i')
    localsize = np.array(swarmdata.shape[0], 'i')
    MPI.COMM_WORLD.Allreduce(localsize, datalenmax, op=MPI.MAX)
    datadimmax = np.array(0, 'i')
    localdim = np.array(swarmdata.shape[1], 'i')
    MPI.COMM_WORLD.Allreduce(localdim, datadimmax, op=MPI.MAX)
    datalength = int(datalenmax * uw.mpi.size)

    globalswarm = np.zeros([datalength, datadimmax])
    dataout = np.empty(datalength)

    for i in range(datadimmax):
        datain = np.full([1, datalenmax], np.nan)[0]
        if(i < localdim):
            datain[:localsize] = swarmdata[:localsize, i]
        MPI.COMM_WORLD.Gather(datain, dataout, root=0)
        globalswarm[:, i] = dataout

    uw.mpi.barrier()
    if(uw.mpi.rank == 0):
        globalswarm = globalswarm[~np.isnan(globalswarm)]
        globalswarm = globalswarm.reshape(int(globalswarm.size/datadimmax), datadimmax)

    return globalswarm

def getGlobalMeshVar(mesh, fielddata):
    if(uw.mpi.size == 1):
        return mesh.data, fielddata
    datalength = np.array(0, 'd')
    datalenmax = np.array(0, 'd')
    localsize = np.array(len(mesh.data[:,0]), 'd')
    MPI.COMM_WORLD.Allreduce(localsize, datalenmax, op=MPI.MAX)
    datalenmax = int(datalenmax)
    datalength = int(datalenmax * uw.mpi.size)
    globalmesh = np.zeros([datalength, mesh.dim])
    globalfield = np.zeros([datalength, fielddata.shape[1]])
    dataout= np.empty(datalength)

    localid = range(mesh.nodesLocal)
    for i in range(mesh.dim):
        datain = np.full([1, datalenmax], np.nan)[0]
        datain[localid] = mesh.data[localid, i]
        MPI.COMM_WORLD.Gather(datain, dataout, root=0)
        globalmesh[:, i] = dataout
    for i in range(fielddata.shape[1]):
        datain = np.full([1, datalenmax], np.nan)[0]
        datain[localid] = fielddata[localid, i]
        MPI.COMM_WORLD.Gather(datain, dataout, root=0)
        globalfield[:, i] = dataout

    uw.mpi.barrier()
    if(uw.mpi.rank == 0):
        globalmesh = globalmesh[~np.isnan(globalmesh)]
        globalfield = globalfield[~np.isnan(globalfield)]
        globalmesh=globalmesh.reshape(int(globalmesh.size/mesh.dim), mesh.dim)
        globalfield = globalfield.reshape(int(globalfield.size/fielddata.shape[1]), fielddata.shape[1])
        if(mesh.dim == 2):
            globalfield = globalfield[np.lexsort([globalmesh[:, 0], globalmesh[:, 1]]), :]
            globalmesh = globalmesh[np.lexsort([globalmesh[:, 0], globalmesh[:, 1]]), :]    
        else:
            globalfield = globalfield[np.lexsort([globalmesh[:, 0], globalmesh[:, 1], globalmesh[:, 2]]), :]
            globalmesh = globalmesh[np.lexsort([globalmesh[:, 0], globalmesh[:, 1], globalmesh[:, 2]]), :]

    return globalmesh, globalfield