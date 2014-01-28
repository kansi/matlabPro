function trainMatrix = svm_classify()
    trainM1     = loadData('./dataset/1/');
    trainM2     = loadData('./dataset/2/');
    trainM3     = loadData('./dataset/3/');
    testMatrix  = loadData('./dataset/4/');
    trainMatrix   = [ trainM1 trainM2 trainM3 ];
    [~, tstColNo] = size(testMatrix);
    
    %calculate mean image
    [meanTrainImage, colNo] = calMean(trainMatrix);
    
    % subtract from mean matrix
    A     = trainMatrix - meanTrainImage(:, sum(eye(colNo)));  
    testA = testMatrix - meanTrainImage(:, sum(eye(tstColNo)));
    
    %calculate eigen vectors for traning data
    eigenVec = calEigenVec(A, 25);
    
    [trainFeatureVec, eigenFace] = calFeatureVec(eigenVec, A);
    %testFeatureVec  = calFeatureVec(eigenVec, testA);
    [~, EFcolNo] = size(eigenFace);
    testFeatureVec = cell(1,tstColNo);
    for i=1:tstColNo
        omega = [];
        for j=1:EFcolNo
            omega = [ omega eigenFace(:,j)'*testA(:,i) ];
        end
        testFeatureVec{i}=omega';
    end 
    [~,m] = size(testFeatureVec);
    [~,n] = size(trainFeatureVec);
     
    writeFile('trainData.txt', trainFeatureVec, n );
    writeFile('testData.txt', testFeatureVec, m );
    return

end

function writeFile(filename, data, m)
    fid = fopen(filename,'w');
    for i=1:m
        [r,~] = size(data{i});
        label = i;
        if mod(label,190) == 0
            label = label-1;
        end
        label = ceil(mod(label,190)/5);
        label
        fprintf(fid,'%d ', label);
        
        for j=1:r
            fprintf(fid,'%d:%f ',j, data{i}(j,1));
        end
        fprintf(fid,'\n');
    end    
    fclose(fid);
end


function [c, eigenFace] = calFeatureVec(eigenVec, A)
    [~,x] = size(eigenVec);
    [~,m] = size(A);
    c     = cell(1,m);
    
    eigenFace = A*eigenVec;
    %Normalization
    for i=1:x
        eigenFace(:,i) = eigenFace(:,i)/norm(eigenFace(:,i)); 
    end
    
    % calulate feature vector here
    for i=1:m
        omega = [];
        for j=1:x
            omega = [ omega  eigenFace(:,j)'*A(:,i) ];
        end
        c{i} = omega';
    end
end

function eigenVec = calEigenVec(A, x)
    prod    = A'*A;
    [V, D]  = eig(prod);             % Get eigen values of a'a
    eigenValues = diag(D);           % convert diagonal matric to 1D
    [~, index] = sort(eigenValues, 'descend'); % sort acc to eigenvalues
    % get x max eigen vectors
    newIndex = index(17:x+17);       % get the top x eigen values
    eigenVec = V(:, newIndex);       % get the corresponding eigenvectors
end

function [meanImage, cols] = calMean(matrix)
    [~,cols]  = size(matrix);
    mean      = sum(matrix');
    meanImage = (mean/cols)';
end

function imageMatrix = loadData(path)
    image = dir(strcat(path, '*.pgm'));
    [m,n] = size(image);
    tmp = [];                       % create an empty matrix
    % for loop for all images
    for i=1:m
        %disp(image(i).name)
        a = double(imread(strcat(path, image(i).name), 'pgm'));  % PxQ matrix of a single image
        [p,q] = size(a);            % get the dimensions of the matrix
        a = reshape(a', 1, p*q)';   % convert matrix to column
        tmp = [tmp a];              % concatenate the columns into the new matrix
    end
    imageMatrix = tmp;
end
