function trainMatrix = ROC()
    testMatrix  = loadData('./dataset/1/');
    trainM1     = loadData('./dataset/2/');
    trainM2     = loadData('./dataset/3/');
    trainM3     = loadData('./dataset/4/');
    trainMatrix = [];
    for i=0:37
        trainMatrix = [ trainMatrix trainM1(:,1+5*i:(i+1)*5)];
        trainMatrix = [ trainMatrix trainM2(:,1+5*i:(i+1)*5)];
        trainMatrix = [ trainMatrix trainM3(:,1+5*i:(i+1)*5)];
    end
    % for CMU-PIE data set uncomment the below line
    %[testMatrix, trainMatrix] = loadData2('./CMU-PIE/');
    
    
    [~, tstColNo] = size(testMatrix);
    
    %calculate mean image
    [meanTrainImage, colNo] = calMean(trainMatrix);
    
    % subtract from mean matrix
    A     = trainMatrix - meanTrainImage(:, sum(eye(colNo)));  
    testA = testMatrix - meanTrainImage(:, sum(eye(tstColNo)));
    
    %calculate eigen vectors for traning data
    eigenVec = calEigenVec(A, 100);
    
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
    
    %Calculate threshold value
    thVal=[];
    for i=1:38
        thVal = [ thVal calThreshold( trainFeatureVec, i )];
    end

    for i=1:m
        if norm(testFeatureVec{i}(:,1)) < thVal(ceil(i/5))
            disp('y')
        else
            disp('n')
        end
    end
    return
end

function threshold = calThreshold(trainFV, class)
    [~,n] = size(trainFV);
    samePair  = createSamePair(trainFV(1+(class-1)*15:class*15) );
    notClass  = cell(1,n-15); 
    % create set of FV not in the current class
    for i=1:(class-1)*15
        notClass{i} = trainFV{i};
    end
    for i=1:n-(class)*15
        notClass{i+(class-1)*15} = trainFV{i+class*15};
    end
    diffPair = createDiffPair(trainFV(1+(class-1)*15:class*15), notClass);
    
    % cal dist
    sameDist     = calDistance(samePair);
    diffDist     = calDistance(diffPair);
    totDist      = [ sameDist diffDist ];
    label        = ones(1000,1);
    label(501:1000) = 0;
    %label
    [x, y, t, k, opt] = perfcurve(label, totDist, 1);
    for i=1:1000
        if x(i)==opt(1) & y(i)==opt(2)
            threshold = t(i);
            break;
        end
    end
    %plot(y,x)
    return
end

function dist = calDistance(vector);
    dist  = [];
    [m,~] = size(vector);
    for i=1:m
        %vector{i,1}
        d = pdist2(vector{i,1}',vector{i,2}','euclidean');
        dist = [ dist d ];
    end
    return
end

function pair = createDiffPair(class, notClass)
    [~,m] = size(class);
    [~,n] = size(notClass);
    pair = cell(500,2);
    x=1;
    y=1;
    for i=1:500
        pair{i,1} = class{x};
        pair{i,2} = notClass{y};
        x = mod(x,15)+1;
        y = y+1;
    end
    return
end

function pair = createSamePair(trainFeatureVec)
    [~,n] = size(trainFeatureVec);
    pair = cell(500,2);
    for i=1:n
        for j=1:n
            pair{(i-1)*15+j, 1} = trainFeatureVec{i};
            pair{(i-1)*15+j, 2} = trainFeatureVec{j};
            pair{(i-1)*15+j+225, 1} = trainFeatureVec{i};
            pair{(i-1)*15+j+225, 2} = trainFeatureVec{j};            
        end
    end
    x = 1;
    y = 1;
    for i=451:500
        pair{i, 1} = trainFeatureVec{x};
        pair{i, 2} = trainFeatureVec{y};
        if y==15
            x = x+1;
            y = 1;
        end
        y = y+1;
    end
    return
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

function [testMatrix, trainMatrix] = loadData2(path)
    data = load(strcat(path, 'CMUPIEData.mat'));
    [~,m] = size(data);
    trainMatrix = [];            % create an empty matrix
    testMatrix  = [];            % create an empty matrix
    % for loop for all different labels
    %images = cell(1,68);
    for i=0:67
        range = data(1).CMUPIEData(1+42*i:20+42*i);
        trainMatrix = [ trainMatrix classify(range, 1,5)  ];
        trainMatrix = [ trainMatrix classify(range, 6,10) ];
        trainMatrix = [ trainMatrix classify(range, 11,15)];
        testMatrix =  [ testMatrix  classify(range, 16,20)];
    end
    %testMatrix
    %size(trainMatrix)
    return
end

function matrix = classify(structs,x,y)
    matrix = [];
    for i=x:y
        matrix = [ matrix structs(i).pixels'];
    end
    return
end