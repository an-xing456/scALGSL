suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Matrix))

readH5AD <- function(dir_name, file_name) {
    file_path = paste(dir_name, '/', file_name, '.h5ad', sep='')
    Convert(file_path, 'h5seurat', overwrite=T)
    new_file_path = paste(dir_name, '/', file_name, '.h5seurat', sep='')    
    data = LoadH5Seurat(new_file_path)
    return (data)
}
read_data <- function(path, data_name) {
    # return matrix        
    path = paste(path, '/', data_name, '.mtx', sep='')    
    data = as(readMM(path), 'matrix') # cell * gene
    data = t(data) # gene * cell    
    return (data)
}

read_gene <- function(path, file_name) {
    path = paste(path, '/', file_name, '.csv', sep='')
    gene = as.matrix(read.csv(path))
    return (gene)
}
read_label <- function(path, label_name) {
    #return matrix
    path = paste(path, '/', label_name, '.csv', sep='')
    label = as.matrix(read.csv(path))
    return (label)
}

select_feature <- function(data,label,nf=2000){
    M <- nrow(data); new.label <- label[,1]
    pv1 <- sapply(1:M, function(i){
        mydataframe <- data.frame(y=as.numeric(data[i,]), ig=new.label)
        fit <- aov(y ~ ig, data=mydataframe)
        summary(fit)[[1]][["Pr(>F)"]][1]})
    names(pv1) <- rownames(data)
    pv1.sig <- names(pv1)[order(pv1)[1:nf]]
    egen <- unique(pv1.sig)
    return (egen)
}

normalize_data <- function(data) {
    # median normalization

    data = t(data) # transpose it to become cell * gene
    row_sum = apply(data, 1, sum)
    mean_t = mean(row_sum)
    # The area where the cell expression level is 0 can be ignored. Setting it to 1 indicates no impact
    row_sum[row_sum==0] = 1

    # row_sum is vector，It will broadcast automatically.
    data = data/row_sum * mean_t
    # turn into gene * cell format
    data = t(data)
    return (data)

}

GenerateGraph <- function(Dat1,Dat2,Dat3,Lab1,K,has_auxilary){
    object1 <- CreateSeuratObject(counts=Dat1,project = "1",assay = "Data1",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1,names.delim = "_")

    object2 <- CreateSeuratObject(counts=Dat2,project = "2",assay = "Data2",
                                  min.cells = 0,min.features =0,names.field = 1,
                                  names.delim = "_")
    if(has_auxilary) {
        object3 <- CreateSeuratObject(counts=Dat3,project = "3",assay = "Data3",
                                  min.cells = 0,min.features =0,names.field = 1,
                                  names.delim = "_")
    
    }
    if (has_auxilary) {
        objects <- list(object1,object2, object3)
    } else{
        objects <- list(object1,object2)
    }
    
    
    objects1 <- lapply(objects,function(obj){
        obj <- NormalizeData(obj,verbose=F)

        obj <- FindVariableFeatures(obj,
                                    selection.method = "vst",
                                    nfeatures = 2000,verbose=F)
        obj <- ScaleData(obj,features=rownames(obj),verbose=FALSE)        
        obj <- RunPCA(obj, features=rownames(obj), verbose = FALSE)
            return(obj)
    })
    
    # Inter-data graph
    # This function uses cca by default
    d1.list <- list(objects1[[1]],objects1[[2]])   
    object.nn <- FindIntegrationAnchors(object.list = d1.list, k.anchor=K,verbose=F)
    arc=object.nn@anchors
    d1.arc1=cbind(arc[arc[,4]==1,1],arc[arc[,4]==1,2],arc[arc[,4]==1,3])
    grp1=d1.arc1[d1.arc1[,3]>0,1:2]-1
    
    # Intra-data graph
    d2.list <- list(objects1[[2]],objects1[[2]])    
    d2.nn <- FindIntegrationAnchors(object.list =d2.list,k.anchor=K,verbose=F)
    d2.arc=d2.nn@anchors
    d2.arc1=cbind(d2.arc[d2.arc[,4]==1,1],d2.arc[d2.arc[,4]==1,2],d2.arc[d2.arc[,4]==1,3])
    d2.grp=d2.arc1[d2.arc1[,3]>0,1:2]-1

    # auxilary data graph
    if (has_auxilary) {
        d3.list <- list(objects1[[3]],objects1[[3]])
        d3.nn <- FindIntegrationAnchors(object.list =d3.list,k.anchor=K,verbose=F)
        d3.arc=d3.nn@anchors
        d3.arc1=cbind(d3.arc[d3.arc[,4]==1,1],d3.arc[d3.arc[,4]==1,2],d3.arc[d3.arc[,4]==1,3])
        d3.grp=d3.arc1[d3.arc1[,3]>0,1:2]-1
        final <- list(inteG=grp1,intraG=d2.grp, auxilaryG=d3.grp)
    }
    else {
        final <- list(inteG=grp1,intraG=d2.grp)
    }
    

    return (final)
}


main <- function(ref_data_dir, 
                query_data_dir,
                auxilary_data_dir,                
                ref_save_path,                 
                query_save_path,                                
                auxilary_save_path,
                has_auxilary){
    
    ref_data = read_data(ref_data_dir, 'ref_data_middle')
    query_data = read_data(query_data_dir, 'query_data_middle')
            
    ref_label = read_label(ref_data_dir, 'ref_label_middle')
    ref_gene = read_gene(ref_data_dir, 'ref_gene_middle')
    query_gene = read_gene(query_data_dir, 'query_gene_middle')
    
    rownames(ref_data) = ref_gene
    rownames(query_data) = query_gene

    if (has_auxilary){
        auxilary_data = read_data(auxilary_data_dir, 'auxilary_data_middle')    
        auxilary_gene = read_gene(auxilary_data_dir, 'auxilary_gene_middle')
        rownames(auxilary_data) = auxilary_gene
    }
                
    # gene intersection
    if (has_auxilary){
        inter_genes = intersect(intersect(rownames(ref_data), rownames(query_data)), rownames(auxilary_data))
        auxilary_data = auxilary_data[inter_genes, ]
    } else {
        inter_genes = intersect(rownames(ref_data), rownames(query_data))
    }    
    ref_data = ref_data[inter_genes, ]
    query_data = query_data[inter_genes, ]
    
    # gene selection
    print(dim(ref_data))
    print(dim(ref_label))
    
    sel.features <- select_feature(ref_data, ref_label)
    sel.ref_data = ref_data[sel.features, ]
    sel.query_data = query_data[sel.features, ]
    
            
    # Norm: gene * cell
    norm.ref_data = normalize_data(sel.ref_data)
    norm.query_data = normalize_data(sel.query_data)
    
    
    # colnames can be anything
    colnames(norm.ref_data) = paste("cell_", 1:ncol(norm.ref_data), sep = "")
    colnames(norm.query_data) = paste("cell_", 1:ncol(norm.query_data), sep = "")
    
    # save selected genes
    write.csv(sel.features, file=paste(paste(base_path, 'data' , 'selected_genes_middle.csv', sep='/')))
        
    norm.auxilary_data = NULL
    # Add the judgment of the auxilary data logic. Here, it is considered that the cross platform does not have auxilaray data
    if (has_auxilary) {
        sel.auxilary_data = auxilary_data[sel.features, ]
        norm.auxilary_data = normalize_data(sel.auxilary_data)
        colnames(norm.auxilary_data) = paste("cell_", 1:ncol(norm.auxilary_data), sep = "")        
        s.norm.auxilary_data <- Matrix(norm.auxilary_data, sparse = TRUE)
        writeMM(s.norm.auxilary_data, auxilary_save_path)    
    }
    
    graphs <- suppressWarnings(GenerateGraph(Dat1=norm.ref_data,Dat2=norm.query_data,Dat3=norm.auxilary_data,
                                                 Lab1=ref_label,K=10,has_auxilary
                                                 ))


    write.csv(graphs[[1]],file=paste(paste(base_path, 'data' , 'inter_graph.csv', sep='/')), quote=F,row.names=T)
    write.csv(graphs[[2]],file=paste(paste(base_path, 'data' , 'intra_graph.csv', sep='/')),quote=F,row.names=T)        
    if (has_auxilary) {
        write.csv(graphs[[3]],file=paste(paste(base_path, 'data' , 'auxilary_graph.csv', sep='/')),quote=F,row.names=T)    
    }
    
    # save sparse matrix
    s.norm.ref_data <- Matrix(norm.ref_data, sparse = TRUE)
    s.norm.query_data <- Matrix(norm.query_data, sparse = TRUE)    
    
    writeMM(s.norm.ref_data, ref_save_path)
    writeMM(s.norm.query_data, query_save_path)    
    
}


args = commandArgs(trailingOnly = TRUE)
base_path = args[[1]]
has_auxilary = args[[2]]

ref_data_dir = paste(base_path, 'raw_data', sep='/')
query_data_dir = paste(base_path, 'raw_data', sep='/')
auxilary_data_dir = paste(base_path, 'raw_data', sep='/')


ref_save_path = paste(base_path, 'data', 'afterNorm_ref_data_middle.mtx',sep='/')
query_save_path = paste(base_path, 'data', 'afterNorm_query_data_middle.mtx', sep='/')
auxilary_save_path = paste(base_path, 'data', 'afterNorm_auxilary_data_middle.mtx', sep='/')

print("Path is")
print(base_path)
main(ref_data_dir, query_data_dir, auxilary_data_dir, ref_save_path, query_save_path, auxilary_save_path, has_auxilary)
